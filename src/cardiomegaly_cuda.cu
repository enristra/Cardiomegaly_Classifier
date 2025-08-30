#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>  
#include <cuda_runtime.h>
#include "cnpy.h"


using namespace std;
namespace fs = std::filesystem;

static int H = 224;
static int W = 224;
static int P = H*W;

static int BLOCK_SIZE = 256;
static int CARDIOMEGALY_ID= 1;

#define CUDA_CHECK(expr) do {                                     \
    cudaError_t _err = (expr);                                    \
    if (_err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(_err));    \
        exit(1);                                                  \
    }                                                             \
} while(0)




//--------KERNELS--------------

//--- (1) Naive baseline: ogni thread fa atomicAdd sullo score per i pixel che elabora
//focus su correttezza funzionale

__global__ void dot_kernel_naive_atomic(const float* __restrict__ images, const float* __restrict__ weights, float* __restrict__ scores, int pixels_per_img) {
    
    const int img = blockIdx.x;
    const int tid = threadIdx.x;
    const float* img_ptr = images + (size_t)img * pixels_per_img;

    //Stride sui pixel di un'immagine
    for (int i = tid; i < pixels_per_img; i += blockDim.x) {

      float prod = img_ptr[i]* weights[i];
      atomicAdd(&scores[img], prod);
    }

}



//--- (2) Baseline con riduzione in shared: più veloce
__global__ void dot_kernel_reduce_shared(const float * __restrict__ images, const float* __restrict__ weights, float* __restrict__ scores, int pixels_per_img){
   
    extern __shared__ float sdata[];
    //ogni blocco si occupa di un'immagine
    const int img = blockIdx.x;
    const int tid = threadIdx.x;

    const float* img_ptr = images + (size_t)img * pixels_per_img;

    float acc = 0.0f;
    //Ogni thread somma una sotto-sequenza di pixel (block-stride)
    for (int i =tid; i<pixels_per_img; i += blockDim.x){
        // fmaf(ai, bi, acc) = acc + ai*bi (accumulo FP32 ragionevole)
        acc = fmaf(img_ptr[i], weights[i], acc);
    }
    sdata[tid] = acc;
    __syncthreads();

    //riduzione ad albero in shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>=1){
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid == 0) scores[img] = sdata[0];

}

// I/O (cnpy)

static vector<string> list_npy_images(const string& dir){
    vector<string> files;
    for (auto& p : fs::directory_iterator(dir)){
        if (!p.is_regular_file()) continue;
        auto path = p.path().string();
        if (p.path().extension()==".npy") files.push_back(path);
    }
    sort(files.begin(), files.end()); // ordinamento deterministico
    return files;
}


// carica pesi, converti in float32 e normalizza a somma=1
static vector<float> load_and_normalize_weights(const string& path){
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(float)){
        fprintf(stderr,"Pesi: atteso float32\n"); exit(1);
    }
    const size_t n = arr.num_vals;
    if (n != (size_t)P){
        fprintf(stderr,"Pesi: size %zu != P=%d\n", n, P); exit(1);
    }
    vector<float> w(P);
    memcpy(w.data(), arr.data<float>(), P*sizeof(float));
    double s = 0.0;
    for (auto v : w) s += v;
    if (abs(s) < 1e-12) { fprintf(stderr,"Pesi: somma zero\n"); exit(1); }
    const float invs = 1.0f / (float)s;
    for (auto& v : w) v *= invs;
    return w;
}


// carica singola immagine .npy (uint8 o float32), appiattita e normalizzata in [0,1]
static void load_image_flatten_01(const string& path, float* out /*size P*/){
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.num_vals != (size_t)P){
        size_t expect1 = (size_t)H*(size_t)W;
        if (arr.num_vals != expect1){
            fprintf(stderr,"Immagine %s: size %zu non compatibile con 224x224\n",
                    path.c_str(), arr.num_vals);
            exit(1);
        }
    }
    if (arr.word_size == sizeof(unsigned char)){
        const unsigned char* p = arr.data<unsigned char>();
        for (int i=0;i<P;++i) out[i] = (float)p[i] / 255.0f;
    } else if (arr.word_size == sizeof(float)) {
        const float* p = arr.data<float>();
        memcpy(out, p, P*sizeof(float));
    } else {
        fprintf(stderr,"Immagine %s: dtype non supportato\n", path.c_str());
        exit(1);
    }
}

// labels: atteso (M, 14) o (M,), usa solo i primi N items (M >= N). Ritorna y.size()==N.
static vector<int> load_labels_binary(const string& path, int N){
    cnpy::NpyArray arr = cnpy::npy_load(path);
    vector<int> y(N, 0);

    // Caso 1: shape (M,)
    if (arr.shape.size() == 1) {
        size_t M = arr.shape[0];
        if (M < (size_t)N) {
            fprintf(stderr,"labels: M=%zu < N=%d (mancano label)\n", M, N);
            exit(1);
        }
        if (arr.word_size == sizeof(unsigned char)) {
            const unsigned char* p = arr.data<unsigned char>();
            for (int i=0;i<N;++i) y[i] = (int)p[i];
        } else if (arr.word_size == sizeof(float)) {
            const float* p = arr.data<float>();
            for (int i=0;i<N;++i) y[i] = (int)lround(p[i]);
        } else {
            fprintf(stderr,"labels: dtype non supportato\n"); exit(1);
        }
    }
    // Caso 2: shape (M, C)
    else if (arr.shape.size() == 2) {
        size_t M = arr.shape[0];
        size_t C = arr.shape[1];
        if (M < (size_t)N) {
            fprintf(stderr,"labels: M=%zu < N=%d (mancano label)\n", M, N);
            exit(1);
        }
        size_t col = 0;
        if (C == 1) col = 0;
        else {
            if (CARDIOMEGALY_ID >= (int)C) {
                fprintf(stderr,"labels: col cardiomegalia=%d fuori range C=%zu\n",
                        CARDIOMEGALY_ID, C);
                exit(1);
            }
            col = (size_t)CARDIOMEGALY_ID;
        }

        if (arr.word_size == sizeof(unsigned char)) {
            const unsigned char* p = arr.data<unsigned char>();
            for (int i=0;i<N;++i) y[i] = (int)p[(size_t)i*C + col];
        } else if (arr.word_size == sizeof(float)) {
            const float* p = arr.data<float>();
            for (int i=0;i<N;++i) y[i] = (int)lround(p[(size_t)i*C + col]);
        } else {
            fprintf(stderr,"labels: dtype non supportato\n"); exit(1);
        }
    }
    else {
        fprintf(stderr,"labels: shape non supportata (ndim=%zu)\n", arr.shape.size());
        exit(1);
    }

    for (auto& v : y) v = (v ? 1 : 0);
    return y;
}

// METRICHE

struct Metrics { double acc=0, sens=0, spec=0, prec=0; long TP=0,TN=0,FP=0,FN=0; };

static Metrics compute_metrics(const vector<int>& y, const vector<int>& yhat){
    Metrics m; size_t N=y.size();
    for (size_t i=0;i<N;++i){
        int t=y[i], p=yhat[i];
        if (t==1 && p==1) ++m.TP;
        else if (t==0 && p==0) ++m.TN;
        else if (t==0 && p==1) ++m.FP;
        else if (t==1 && p==0) ++m.FN;
    }
    const double eps=1e-12;
    m.acc  = double(m.TP+m.TN)/max<size_t>(1,N);
    m.sens = double(m.TP)/max<double>(eps, m.TP+m.FN);
    m.spec = double(m.TN)/max<double>(eps, m.TN+m.FP);
    m.prec = double(m.TP)/max<double>(eps, m.TP+m.FP);
    return m;
}


// Soglia auto (Youden): sweep su score ordinati
static float auto_threshold_youden(const vector<float>& scores, const vector<int>& y){
    // Soglie candidate = valori unici degli score
    vector<float> thr = scores;
    sort(thr.begin(), thr.end());
    thr.erase(unique(thr.begin(), thr.end()), thr.end());
    if (thr.empty()) return 0.5f;

    auto youden_at = [&](float t){
        long TP=0,TN=0,FP=0,FN=0;
        const size_t N = scores.size();
        for (size_t i=0;i<N;++i){
            int p = (scores[i] > t) ? 1 : 0;  // <-- stessa regola usata poi
            int a = y[i];
            if (p==1 && a==1) ++TP;
            else if (p==1 && a==0) ++FP;
            else if (p==0 && a==1) ++FN;
            else ++TN;
        }
        double sens = (TP+FN) ? double(TP)/double(TP+FN) : 0.0;
        double spec = (TN+FP) ? double(TN)/double(TN+FP) : 0.0;
        return sens + spec - 1.0; // Youden's J
    };

    float best_t = thr.front();
    double best_J = -1e9;
    for (float t : thr){
        double J = youden_at(t);
        if (J > best_J){ best_J = J; best_t = t; }
    }
    return best_t;
}


// CLI
struct Args{
    string images_dir;
    string weights_path;
    string labels_path;
    string out_csv="results.csv";
    string out_metrics="metrics.txt";
    int limit =-1;
    float threshold =0.0f;
    bool auto_th= false;
    bool have_th=false;
    string kernel = "reduce"; //naive || reduce
    int block = BLOCK_SIZE;
};

static void print_usage(const char* prog){
    fprintf(stderr,
        "Uso:\n"
        "  %s --images-dir <dir> --weights <weights.npy> [--labels <labels.npy>]\n"
        "     [--limit N] [--out-csv file.csv] [--out-metrics file.txt]\n"
        "     [--threshold T | --auto-th] [--kernel reduce|naive] [--block B]\n", prog);
}

static Args parse_args(int argc, char** argv){
    Args a;
    
    for (int i=1;i<argc;++i){
        string k = argv[i];
        auto need = [&](int m){ if (i+m>=argc){ print_usage(argv[0]); exit(1);} };
        if (k=="--images-dir"){ need(1); a.images_dir = argv[++i]; }
        else if (k=="--weights"){ need(1); a.weights_path = argv[++i]; }
        else if (k=="--labels"){ need(1); a.labels_path = argv[++i]; }
        else if (k=="--limit"){ need(1); a.limit = stoi(argv[++i]); }
        else if (k=="--out-csv"){ need(1); a.out_csv = argv[++i]; }
        else if (k=="--out-metrics"){ need(1); a.out_metrics = argv[++i]; }
        else if (k=="--threshold"){ need(1); a.threshold = stof(argv[++i]); a.have_th = true; }
        else if (k=="--auto-th"){ a.auto_th = true; }
        else if (k=="--kernel"){ need(1); a.kernel = argv[++i]; }
        else if (k=="--block"){ need(1); a.block = stoi(argv[++i]); }
        else { fprintf(stderr, "Argomento sconosciuto: %s\n", k.c_str()); print_usage(argv[0]); exit(1); }
    }

    if (a.images_dir.empty() || a.weights_path.empty()){
        fprintf(stderr, "Richiesti --images-dir e --weights\n"); print_usage(argv[0]); exit(1);
    }
    if (a.kernel!="reduce" && a.kernel!="naive"){
        fprintf(stderr, "--kernel deve essere reduce|naive\n"); exit(1);
    }
    if (a.block<=0 || a.block>1024){
        fprintf(stderr, "--block fuori range (1..1024)\n"); exit(1);
    }
    return a;

}

int main(int argc, char** argv) {
   Args args = parse_args(argc, argv);


     // 1) Lista immagini
    vector<string> files = list_npy_images(args.images_dir);
    if (files.empty()){ fprintf(stderr,"Nessuna .npy in %s\n", args.images_dir.c_str()); return 1; }
    if (args.limit>0 && (int)files.size()>args.limit) files.resize(args.limit);
    const int N = (int)files.size();
    printf("Trovate %d immagini in %s\n", N, args.images_dir.c_str());

    // 2) Pesi (normalizzati)
    vector<float> weights = load_and_normalize_weights(args.weights_path);

    // 3) Carica immagini in (N,P) float32 [0,1]
    vector<float> h_images((size_t)N * P);
    for (int i=0;i<N;++i){
        load_image_flatten_01(files[i], h_images.data() + (size_t)i*P);
    }


    // 4) Labels
    vector<int> y;
    if (!args.labels_path.empty()){
        y = load_labels_binary(args.labels_path, N);
    }

// 5) GPU: calcolo scores[N]
    float *d_images=nullptr, *d_weights=nullptr, *d_scores=nullptr;
    size_t bytes_img = (size_t)N*P*sizeof(float);
    size_t bytes_w   = (size_t)P*sizeof(float);
    size_t bytes_out = (size_t)N*sizeof(float);
    
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);


    CUDA_CHECK(cudaMalloc((void**)&d_images, bytes_img));
    CUDA_CHECK(cudaMalloc((void**)&d_weights, bytes_w));
    CUDA_CHECK(cudaMalloc((void**)&d_scores, bytes_out));
    CUDA_CHECK(cudaMemcpy(d_images, h_images.data(), bytes_img, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), bytes_w, cudaMemcpyHostToDevice));

    dim3 grid(N), block(args.block);
    size_t shmem = (size_t)args.block * sizeof(float);
    cudaEventRecord(ev_start);
    if (args.kernel=="naive"){
        CUDA_CHECK(cudaMemset(d_scores, 0, bytes_out));
        dot_kernel_naive_atomic<<<grid, block>>>(d_images, d_weights, d_scores, P);
    } else {
        dot_kernel_reduce_shared<<<grid, block, shmem>>>(d_images, d_weights, d_scores, P);
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);


    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    printf("[GPU] kernel time: %.3f ms\n", gpu_ms);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);


    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<float> scores(N);
    CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, bytes_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_scores));

    // 6) Soglia
    float th = args.threshold;
    if (!y.empty()){
        if (args.auto_th || !args.have_th){
            th = auto_threshold_youden(scores, y);
            printf("Auto soglia (Youden): %.6f\n", th);
        } else {
            printf("Soglia fissa: %.6f\n", th);
        }
    }

    // 7) Predizioni + metriche
    vector<int> pred(N, 0);
    for (int i=0;i<N;++i) pred[i] = (scores[i] > th) ? 1 : 0;

    Metrics m;
    if (!y.empty()) m = compute_metrics(y, pred);

    // 8) Output CSV per-immagine
    {
        ofstream csv(args.out_csv);
        csv << "filename,score,pred,actual,correct\n";
        for (int i=0;i<N;++i){
            int actual = y.empty()? -1 : y[i];
            int correct = y.empty()? -1 : (pred[i]==actual ? 1 : 0);
            csv << fs::path(files[i]).filename().string() << ","
                << setprecision(8) << scores[i] << ","
                << pred[i] << ","
                << actual << ","
                << correct << "\n";
        }
        printf("Salvato CSV: %s\n", args.out_csv.c_str());
    }

    // 9) Output metriche aggregate
    if (!y.empty()){
        ofstream txt(args.out_metrics);
        txt << fixed << setprecision(6);
        txt << "threshold: " << th << "\n\n";
        txt << "confusion_matrix:\n";
        txt << "TN="<<m.TN<<"  FP="<<m.FP<<"\n";
        txt << "FN="<<m.FN<<"  TP="<<m.TP<<"\n\n";
        txt << "accuracy="<<m.acc<<"\n";
        txt << "sensitivity="<<m.sens<<"\n";
        txt << "specificity="<<m.spec<<"\n";
        txt << "precision="<<m.prec<<"\n";
        printf("Salvato metriche: %s\n", args.out_metrics.c_str());
    }

    // 10) Stampa riassunto
    printf("Kernel=%s  Block=%d  N=%d  P=%d\n",
           args.kernel.c_str(), args.block, N, P);
    if (!y.empty()){
        printf("Acc=%.4f  Sens=%.4f  Spec=%.4f  Prec=%.4f  (th=%.6f)\n",
               m.acc, m.sens, m.spec, m.prec, th);
    }
    return 0;
}
