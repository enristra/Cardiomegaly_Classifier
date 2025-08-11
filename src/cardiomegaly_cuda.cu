#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <dirent.h>   // opendir/readdir/closedir
#include <sys/stat.h> // stat
#include "cnpy.h"

using namespace std;

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call);  \
    if (err != cudaSuccess) {  \
        cerr << "CUDA error " << cudaGetErrorString(err) \
             << " at " << __FILE__ << ":" << __LINE__ << endl; \
        std::exit(1); \
    } \
} while(0)

static const int H = 224;
static const int W = 224;
static const int P = H * W; // 50176

// --------------------------------- UTILS ---------------------------------

static bool file_exists(const string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

static bool dir_exists(const string& path) {
    struct stat st;
    return (stat(path.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
}

static bool starts_with(const string& s, const char* prefix) {
    size_t lp = strlen(prefix);
    return s.size() >= lp && strncmp(s.c_str(), prefix, lp) == 0;
}

static bool ends_with(const string& s, const char* suffix) {
    size_t ls = s.size(), l = strlen(suffix);
    return (ls >= l) && (s.compare(ls - l, l, suffix) == 0);
}

// ------------------------------- CUDA KERNEL ------------------------------

__global__ void dot_per_image_kernel(const float* __restrict__ images,
                                     const float* __restrict__ weights,
                                     float* __restrict__ scores,
                                     int pixels_per_img) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int img_idx = blockIdx.x;
    const float* img = images + img_idx * pixels_per_img;

    float sum = 0.0f;
    for (int i = tid; i < pixels_per_img; i += blockDim.x) {
        sum += img[i] * weights[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = (blockDim.x >> 1); s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) scores[img_idx] = sdata[0];
}

// ------------------------------- I/O HELPERS -----------------------------

static vector<float> load_weights_or_die(const string& path) {
    cout << "[W] Carico pesi: " << path << endl;
    cnpy::NpyArray arr = cnpy::npy_load(path);

    if (arr.shape.size() != 2 || arr.shape[0] != (size_t)H || arr.shape[1] != (size_t)W) {
        throw runtime_error("weights.npy deve avere shape 224x224");
    }

    vector<float> w(P);
    if (arr.word_size == sizeof(float)) {
        const float* p = arr.data<float>();
        copy(p, p + P, w.begin());
    } else if (arr.word_size == sizeof(double)) {
        const double* p = arr.data<double>();
        for (int i = 0; i < P; ++i) w[i] = (float)p[i];
    } else {
        throw runtime_error("Tipo pesi non supportato (attesi float32 o float64)");
    }

    // opzionale: normalizzazione somma=1
    double s = 0.0;
    for (int i = 0; i < P; ++i) s += w[i];
    if (s != 0.0) {
        for (int i = 0; i < P; ++i) w[i] = (float)(w[i] / s);
    }

    cout << "[W] OK (" << H << "x" << W << ")\n";
    return w;
}

static void save_scores_npy(const string& path, const vector<float>& scores) {
    vector<size_t> shape(1);
    shape[0] = (size_t)scores.size();
    cnpy::npy_save(path, scores.data(), shape, "w");
    cout << "[OUT] Salvato " << path << " (" << scores.size() << " elementi)\n";
}

// Carica batch (N,50176 float32) o (N,224,224 uint8/float32), altrimenti directory image_*.npy
// max_images > 0 limita il numero di immagini
static vector<float> load_images_batch_or_dir(const string& batch_path,
                                              const string& images_dir,
                                              int max_images,
                                              int& N_out) {
    // 1) batch file
    if (!batch_path.empty() && file_exists(batch_path)) {
        cout << "[IMG] Provo batch: " << batch_path << endl;
        cnpy::NpyArray arr = cnpy::npy_load(batch_path);

        if (arr.shape.size() == 2 && arr.shape[1] == (size_t)P) {
            if (arr.word_size != sizeof(float)) throw runtime_error("Batch (N x 50176) deve essere float32");
            int N_all = (int)arr.shape[0];
            int N_lim = (max_images > 0 && max_images < N_all) ? max_images : N_all;
            const float* p = arr.data<float>();
            N_out = N_lim;

            vector<float> out(N_out * P);
            copy(p, p + (size_t)N_out * P, out.begin());
            cout << "[IMG] Batch (N,P). Uso N=" << N_out << "\n";
            return out;
        }
        if (arr.shape.size() == 3 && arr.shape[1] == (size_t)H && arr.shape[2] == (size_t)W) {
            int N_all = (int)arr.shape[0];
            int N_lim = (max_images > 0 && max_images < N_all) ? max_images : N_all;
            N_out = N_lim;

            vector<float> out(N_out * P);
            if (arr.word_size == sizeof(uint8_t)) {
                const uint8_t* p = arr.data<uint8_t>();
                for (int n = 0; n < N_out; ++n) {
                    const uint8_t* img = p + (size_t)n * P;
                    float* dst = out.data() + (size_t)n * P;
                    for (int i = 0; i < P; ++i) dst[i] = img[i] / 255.0f;
                }
            } else if (arr.word_size == sizeof(float)) {
                const float* p = arr.data<float>();
                for (int n = 0; n < N_out; ++n) {
                    const float* img = p + (size_t)n * P;
                    float* dst = out.data() + (size_t)n * P;
                    copy(img, img + P, dst);
                }
            } else {
                throw runtime_error("Batch (N,224,224) deve essere uint8 o float32");
            }
            cout << "[IMG] Batch (N,224,224). Uso N=" << N_out << "\n";
            return out;
        }

        throw runtime_error("Batch shape inattesa. Attesi (N,50176) o (N,224,224).");
    }

    // 2) directory
    cout << "[IMG] Batch non trovato. Provo directory: " << images_dir << endl;
    if (!dir_exists(images_dir)) throw runtime_error("Directory immagini non esiste: " + images_dir);

    vector<string> files;
    DIR* d = opendir(images_dir.c_str());
    if (!d) throw runtime_error("Impossibile aprire la directory: " + images_dir);

    struct dirent* de;
    while ((de = readdir(d)) != NULL) {
        string name = de->d_name;
        if (starts_with(name, "image_") && ends_with(name, ".npy")) {
            files.push_back(images_dir + "/" + name);
        }
    }
    closedir(d);

    sort(files.begin(), files.end());
    if (files.empty()) throw runtime_error("Nessuna immagine .npy trovata.");

    if (max_images > 0 && (int)files.size() > max_images) files.resize((size_t)max_images);

    int N = (int)files.size();
    cout << "[IMG] Carico " << N << " file .npy..." << endl;

    vector<float> all((size_t)N * P);

    for (int idx = 0; idx < N; ++idx) {
        if ((idx + 1) % 500 == 0) cout << "  [IMG] Caricate " << (idx + 1) << " immagini" << endl;

        cnpy::NpyArray a = cnpy::npy_load(files[idx]);
        float* dst = all.data() + (size_t)idx * P;

        if (a.shape.size() == 2 && a.shape[0] == (size_t)H && a.shape[1] == (size_t)W) {
            if (a.word_size == sizeof(uint8_t)) {
                const uint8_t* p = a.data<uint8_t>();
                for (int i = 0; i < P; ++i) dst[i] = p[i] / 255.0f;
            } else if (a.word_size == sizeof(float)) {
                const float* p = a.data<float>();
                copy(p, p + P, dst);
            } else {
                throw runtime_error("Formato immagine non supportato (224x224)");
            }
        }
        else if (a.shape.size() == 1 && a.shape[0] == (size_t)P) {
            if (a.word_size != sizeof(float)) throw runtime_error("Immagine flatten deve essere float32");
            const float* p = a.data<float>();
            copy(p, p + P, dst);
        }
        else {
            throw runtime_error("Dimensione inattesa in " + files[idx]);
        }
    }

    N_out = N;
    cout << "[IMG] OK. N=" << N_out << "\n";
    return all;
}

// baseline CPU opzionale
static vector<float> cpu_dot_scores(const vector<float>& imgs,
                                    const vector<float>& w,
                                    int N) {
    vector<float> s(N);
    for (int n = 0; n < N; ++n) {
        const float* img = imgs.data() + (size_t)n * P;
        double acc = 0.0;
        for (int i = 0; i < P; ++i) acc += img[i] * w[i];
        s[n] = (float)acc;
    }
    return s;
}

// ------------------------------------ MAIN ------------------------------------

int main(int argc, char** argv) {
    try {
        string weights_path = "../data/weights/static_weights_224x224.npy";
        string images_batch = "../data/images_batch_224x224.npy";
        string images_dir   = "../data/ChestMNIST_Images";
        string out_scores   = "../results/scores_gpu.npy";
        int block_size = 256;   // potenza di 2 consigliata
        int max_images = -1;    // -1 = nessun limite
        bool run_cpu = true;

        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "--weights") && i + 1 < argc) weights_path = argv[++i];
            else if (!strcmp(argv[i], "--batch") && i + 1 < argc) images_batch = argv[++i];
            else if (!strcmp(argv[i], "--images") && i + 1 < argc) images_dir = argv[++i];
            else if (!strcmp(argv[i], "--out")    && i + 1 < argc) out_scores = argv[++i];
            else if (!strcmp(argv[i], "--bs")     && i + 1 < argc) block_size = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--max-img")&& i + 1 < argc) max_images = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--no-cpu")) run_cpu = false;
        }

        // carica
        vector<float> weights = load_weights_or_die(weights_path);

        int N = 0;
        vector<float> images = load_images_batch_or_dir(images_batch, images_dir, max_images, N);
        cout << "[SET] N=" << N << ", P=" << P << ", block_size=" << block_size << endl;

        // CPU baseline
        vector<float> cpu_scores;
        double cpu_ms = 0.0;
        if (run_cpu) {
            auto t0 = chrono::high_resolution_clock::now();
            cpu_scores = cpu_dot_scores(images, weights, N);
            auto t1 = chrono::high_resolution_clock::now();
            cpu_ms = chrono::duration<double, milli>(t1 - t0).count();
            cout << "[CPU] dot total: " << cpu_ms << " ms" << endl;
        }

        // GPU
        float *d_images = NULL, *d_weights = NULL, *d_scores = NULL;
        CHECK_CUDA(cudaMalloc((void**)&d_images,  (size_t)N * P * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_weights, (size_t)P * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_scores,  (size_t)N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_images,  images.data(),  (size_t)N * P * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_weights, weights.data(), (size_t)P * sizeof(float),     cudaMemcpyHostToDevice));

        dim3 grid((unsigned int)N);
        dim3 block((unsigned int)block_size);
        size_t shmem = (size_t)block_size * sizeof(float);

        cudaEvent_t ev_start, ev_stop;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));

        CHECK_CUDA(cudaEventRecord(ev_start));
        dot_per_image_kernel<<<grid, block, shmem>>>(d_images, d_weights, d_scores, P);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        float gpu_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
        cout << "[GPU] kernel time: " << gpu_ms << " ms" << endl;

        vector<float> scores(N);
        CHECK_CUDA(cudaMemcpy(scores.data(), d_scores, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));

        if (run_cpu) {
            double mae = 0.0, maxae = 0.0;
            for (int i = 0; i < N; ++i) {
                double ae = fabs((double)scores[i] - (double)cpu_scores[i]);
                mae += ae;
                if (ae > maxae) maxae = ae;
            }
            if (N > 0) mae /= N;
            cout << "[CHK] CPU vs GPU -> MAE: " << mae << "  maxAE: " << maxae << endl;
        }

        save_scores_npy(out_scores, scores);

        CHECK_CUDA(cudaFree(d_images));
        CHECK_CUDA(cudaFree(d_weights));
        CHECK_CUDA(cudaFree(d_scores));
        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_stop));

        cout << "[OK] Fine" << endl;
        return 0;

    } catch (const exception& e) {
        cerr << "Errore: " << e.what() << endl;
        return 1;
    }
}
