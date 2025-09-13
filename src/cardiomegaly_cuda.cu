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

#ifndef TILE_X
#define TILE_X 32
#endif
#ifndef TILE_Y
#define TILE_Y 8
#endif


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

// Versione ottimizzata con caricamenti float4 + riduzione in shared
// Somma pesata con load vettoriali, senza reinterpret_cast nel kernel
// Somma pesata con load vettoriali, senza reinterpret_cast nel kernel
__global__ void dot_kernel_reduce_vec4(const float4* __restrict__ images4,
                                             const float4* __restrict__ weights4,
                                             float* __restrict__ scores,
                                             int pixels4_per_img)   // = P/4
{
    extern __shared__ float sdata[];
    const int img = blockIdx.x;
    const int tid = threadIdx.x;

    const float4* img4 = images4 + (size_t)img * pixels4_per_img;

    // ILP semplice: 4 accumulatori
    float acc0=0.f, acc1=0.f, acc2=0.f, acc3=0.f;

    #pragma unroll 4
    for (int i4 = tid; i4 < pixels4_per_img; i4 += blockDim.x) {
        float4 a = img4[i4];
        float4 b = weights4[i4];
        acc0 = fmaf(a.x, b.x, acc0);
        acc1 = fmaf(a.y, b.y, acc1);
        acc2 = fmaf(a.z, b.z, acc2);
        acc3 = fmaf(a.w, b.w, acc3);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);

    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) scores[img] = sdata[0];
}


// images: [N][H*W] contigue, weights: [H*W] condivisi per tutte le immagini
__global__ void dot_kernel_tile2d_reduce_shared(
    const float* __restrict__ images,
    const float* __restrict__ weights,
    float* __restrict__ scores,
    int H, int W, int stride_img)     // stride_img = H*W
{
    const int img = blockIdx.x;                   // 1 blocco = 1 immagine
    const int tx  = threadIdx.x;                  // [0, TILE_X)
    const int ty  = threadIdx.y;                  // [0, TILE_Y)
    const int tid = ty * blockDim.x + tx;         // [0, TILE_X*TILE_Y)

    const float* __restrict__ img_ptr = images + (size_t)img * stride_img;

    extern __shared__ float s[];
    float* s_img = s;                                              // TILE_X*TILE_Y
    float* s_w   = s_img + (TILE_X * TILE_Y);                      // TILE_X*TILE_Y
    float* s_red = s_w   + (TILE_X * TILE_Y);                      // TILE_X*TILE_Y

    float acc = 0.0f;

    // Scansione in tiles 2D
    for (int y0 = 0; y0 < H; y0 += TILE_Y) {
        for (int x0 = 0; x0 < W; x0 += TILE_X) {

            const int x = x0 + tx;
            const int y = y0 + ty;
            const int lin = ty * TILE_X + tx;

            float img_val = 0.0f, w_val = 0.0f;
            if (x < W && y < H) {
                const int idx = y * W + x;
                img_val = img_ptr[idx];
                w_val   = weights[idx];
            }
            s_img[lin] = img_val;
            s_w[lin]   = w_val;

            __syncthreads();

            // Prodotto del proprio elemento del tile
            acc += s_img[lin] * s_w[lin];

            __syncthreads(); // prima di ri-sovrascrivere i tile
        }
    }

    // Riduzione nel blocco (solo shared, niente warp shuffle)
    s_red[tid] = acc;
    __syncthreads();

    for (int offset = (blockDim.x * blockDim.y) >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) s_red[tid] += s_red[tid + offset];
        __syncthreads();
    }

    if (tid == 0) scores[img] = s_red[0];
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



void launch_tile2d(
    const float* d_images,
    const float* d_weights,
    float* d_scores,
    int N_images, int H, int W,
    cudaStream_t stream = 0)
{
    dim3 block(TILE_X, TILE_Y);     // 32x8 = 256 thread (buona coalescenza orizzontale)
    dim3 grid(N_images);            // 1 blocco per immagine
    const int stride_img = H * W;

    size_t shmem = 3 * TILE_X * TILE_Y * sizeof(float); // img + w + buffer riduzione
    dot_kernel_tile2d_reduce_shared<<<grid, block, shmem, stream>>>(
        d_images, d_weights, d_scores, H, W, stride_img);
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
        "     [--threshold T | --auto-th]\n"
        "     [--kernel reduce|naive|vec4|tile2d] [--block B]\n"
        "Note: per tile2d il block viene ignorato; usa TILE_X/TILE_Y a compile-time (es. -DTILE_X=32 -DTILE_Y=8)\n",
        prog);
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
    if (a.kernel!="reduce" && a.kernel!="naive" && a.kernel!="vec4" && a.kernel!="tile2d"){
    fprintf(stderr, "--kernel deve essere reduce|naive|vec4|tile2d\n");
    exit(1);
}
    if (a.block<=0 || a.block>1024){
        fprintf(stderr, "--block fuori range (1..1024)\n"); exit(1);
    }
    return a;

}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (P % 4 != 0) {
        fprintf(stderr, "Errore: P=%d non è multiplo di 4 (richiesto per kernel vec4)\n", P);
        return 1;
    }
    const int P4 = P / 4;

    // --- 1) Image list ---
    vector<string> files = list_npy_images(args.images_dir);
    if (files.empty()) {
        fprintf(stderr, "Nessuna .npy in %s\n", args.images_dir.c_str());
        return 1;
    }

    if (args.limit > 0 && (int)files.size() > args.limit) files.resize(args.limit);
    const int N = (int)files.size();
    printf("Trovate %d immagini in %s\n", N, args.images_dir.c_str());

    // --- 2) Weights (normalized) ---
    vector<float> weights = load_and_normalize_weights(args.weights_path);

    // --- 3) Load host images in (N, P) float32 [0,1] ---
    vector<float> h_images((size_t)N * P);
    for (int i = 0; i < N; ++i) {
        load_image_flatten_01(files[i], h_images.data() + (size_t)i * P);
    }

    // --- 4) Labels ---
    vector<int> y;
    if (!args.labels_path.empty()) {
        y = load_labels_binary(args.labels_path, N);
        // count pos/neg
        int num_pos = 0, num_neg = 0;
        for (int l : y) (l > 0) ? num_pos++ : num_neg++;
        printf("Numero di campioni positivi: %d, negativi: %d\n", num_pos, num_neg);
    }

    // --- 5) GPU: scores calculation ---
    const size_t bytes_img_all = (size_t)N * P * sizeof(float);
    const size_t bytes_w       = (size_t)P * sizeof(float);
    const size_t bytes_out     = (size_t)N * sizeof(float);
    vector<float> scores(N, 0.f);


    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

   // -- 6) Start Kernel
    float gpu_ms = 0.0f;
    dim3 grid(N), block(args.block);
    size_t shmem = (size_t)args.block * sizeof(float);

    if (args.kernel == "naive") {
        float *d_images = nullptr, *d_weights = nullptr, *d_scores = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_images, bytes_img_all));
        CUDA_CHECK(cudaMalloc((void**)&d_weights, bytes_w));
        CUDA_CHECK(cudaMalloc((void**)&d_scores, bytes_out));

        CUDA_CHECK(cudaMemcpy(d_images,  h_images.data(), bytes_img_all, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights, weights.data(),  bytes_w,       cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_scores, 0, bytes_out));

        CUDA_CHECK(cudaEventRecord(ev_start));
        dot_kernel_naive_atomic<<<grid, block>>>(d_images, d_weights, d_scores, P);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
        printf("[GPU] kernel time: %.3f ms\n", gpu_ms);

        CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, bytes_out, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_images));
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_scores));
    }
    else if (args.kernel == "reduce") {
        // --- reduce (shared): float* ---
        float *d_images = nullptr, *d_weights = nullptr, *d_scores = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_images, bytes_img_all));
        CUDA_CHECK(cudaMalloc((void**)&d_weights, bytes_w));
        CUDA_CHECK(cudaMalloc((void**)&d_scores, bytes_out));

        CUDA_CHECK(cudaMemcpy(d_images,  h_images.data(), bytes_img_all, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights, weights.data(),  bytes_w,       cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaFuncSetCacheConfig(dot_kernel_reduce_shared, cudaFuncCachePreferL1));

        CUDA_CHECK(cudaEventRecord(ev_start));
        dot_kernel_reduce_shared<<<grid, block, shmem>>>(d_images, d_weights, d_scores, P);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
        printf("[GPU] kernel time: %.3f ms\n", gpu_ms);

        CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, bytes_out, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_images));
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_scores));
    }
    else if (args.kernel == "vec4") {
        // --- vec4: tipizza direttamente come float4* ---
        float4 *d_images4 = nullptr, *d_weights4 = nullptr;
        float  *d_scores  = nullptr;

       
        CUDA_CHECK(cudaMalloc((void**)&d_images4, (size_t)N * P4 * sizeof(float4)));
        CUDA_CHECK(cudaMalloc((void**)&d_weights4,         P4 * sizeof(float4)));
        CUDA_CHECK(cudaMalloc((void**)&d_scores,  bytes_out));

        CUDA_CHECK(cudaMemcpy(d_images4,  h_images.data(), bytes_img_all, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights4, weights.data(),  bytes_w,       cudaMemcpyHostToDevice));

        // More L1 cache 
        CUDA_CHECK(cudaFuncSetCacheConfig(dot_kernel_reduce_vec4, cudaFuncCachePreferL1));

        CUDA_CHECK(cudaEventRecord(ev_start));
        dot_kernel_reduce_vec4<<<grid, block, shmem>>>(d_images4, d_weights4, d_scores, P4);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
        printf("[GPU] kernel time: %.3f ms\n", gpu_ms);

        CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, bytes_out, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_images4));
        CUDA_CHECK(cudaFree(d_weights4));
        CUDA_CHECK(cudaFree(d_scores));
    }
    else if (args.kernel == "tile2d") {
        // --- tile2d: usa blocchi 2D (TILE_X,TILE_Y) + tiling in shared ---
        float *d_images = nullptr, *d_weights = nullptr, *d_scores = nullptr;

        CUDA_CHECK(cudaMalloc((void**)&d_images, bytes_img_all));
        CUDA_CHECK(cudaMalloc((void**)&d_weights, bytes_w));
        CUDA_CHECK(cudaMalloc((void**)&d_scores, bytes_out));

        CUDA_CHECK(cudaMemcpy(d_images,  h_images.data(), bytes_img_all, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights, weights.data(),  bytes_w,       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_scores, 0, bytes_out));

        // Profilazione
        CUDA_CHECK(cudaEventRecord(ev_start));

        // Lancia tramite il tuo launcher (usa dim3(TILE_X,TILE_Y) internamente)
        launch_tile2d(d_images, d_weights, d_scores, N, H, W /*, stream=0*/);

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
        printf("[GPU] kernel time: %.3f ms\n", gpu_ms);

        CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, bytes_out, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_images));
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_scores));
        }
    else {
        fprintf(stderr, "Unknown Kernel : %s\n", args.kernel.c_str());
        return 1;
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    // --- 7) Stampa min/max/median ---
    float min_score = numeric_limits<float>::max();
    float max_score = numeric_limits<float>::lowest();
    vector<float> scores_copy = scores;
    for (float s : scores) { min_score = min(min_score, s); max_score = max(max_score, s); }
    sort(scores_copy.begin(), scores_copy.end());
    float median_score = scores_copy[scores_copy.size()/2];
    printf("Score min: %.6f, max: %.6f, median: %.6f\n", min_score, max_score, median_score);


    // --- 8) Threshold: fixed/median
    float used_threshold = args.have_th ? args.threshold : median_score;
    printf("Threshold utilizzata: %.6f\n", used_threshold);

    // --- 9) Predictions
    vector<int> pred(N, 0);
    for (int i = 0; i < N; ++i) pred[i] = (scores[i] > used_threshold) ? 1 : 0;

    // --- 10) Metrics ---
    Metrics m;
    if (!y.empty()) m = compute_metrics(y, pred);

  

    // --- 11) Save CSV/TXT
    ofstream txt(args.out_metrics);
    txt << fixed << setprecision(6);
    txt << "threshold: " << used_threshold << "\n\n";
    txt << "confusion_matrix:\n";
    txt << "TN=" << m.TN << "  FP=" << m.FP << "\n";
    txt << "FN=" << m.FN << "  TP=" << m.TP << "\n\n";
    txt << "accuracy="    << m.acc  << "\n";
    txt << "sensitivity=" << m.sens << "\n";
    txt << "specificity=" << m.spec << "\n";
    txt << "precision="   << m.prec << "\n";
    printf("Salvato metriche: %s\n", args.out_metrics.c_str());

     ofstream ofs(args.out_csv);
    ofs << "filename,score,pred,actual\n";
    for (int i = 0; i < N; ++i){
        int a = y.empty() ? -1 : y[i];
        ofs << files[i] << "," << scores[i] << "," << pred[i] << "," << a << "\n";
    }
   printf("Salvato CSV: %s\n", args.out_csv.c_str());

    printf("Kernel=%s  Block=%d  N=%d  P=%d\n", args.kernel.c_str(), args.block, N, P);
    if (!y.empty()) printf("Acc=%.4f  Sens=%.4f  Spec=%.4f  Prec=%.4f  (th=%.6f)\n",
                            m.acc, m.sens, m.spec, m.prec, used_threshold);

    return 0;
}  
