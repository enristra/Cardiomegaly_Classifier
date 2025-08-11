#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "cnpy.h"

namespace fs = std::filesystem;

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call);  \
    if (err != cudaSuccess) {  \
        std::cerr << "CUDA error " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

static constexpr int H = 224;
static constexpr int W = 224;
static constexpr int P = H*W; // 50176

// Kernel: un blocco per immagine, riduzione in shared memory
__global__ void dot_per_image_kernel(const float* __restrict__ images,
                                     const float* __restrict__ weights,
                                     float* __restrict__ scores,
                                     int pixels_per_img) {
    extern __shared__ float sdata[]; // dimensione = blockDim.x
    const int tid = threadIdx.x;
    const int img_idx = blockIdx.x;

    const float* img = images + img_idx * pixels_per_img;

    float sum = 0.0f;
    // ogni thread attraversa i pixel a stride
    for (int i = tid; i < pixels_per_img; i += blockDim.x) {
        // letture coalesced su img; weights è uguale per tutti
        sum += img[i] * weights[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // riduzione in shared
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        scores[img_idx] = sdata[0];
    }
}

// -------- Utils I/O --------
static std::vector<float> load_weights_or_die(const std::string& path) {
    std::cout << "Carico pesi: " << path << std::endl;
    cnpy::NpyArray arr = cnpy::npy_load(path);
    std::cout << "Pesi caricati. Shape: " << arr.shape[0] << "x" << arr.shape[1] << std::endl;

    if (arr.shape.size() != 2 || arr.shape[0] != H || arr.shape[1] != W)
        throw std::runtime_error("weights.npy deve essere 224x224");
    std::vector<float> w;
    if (arr.word_size == sizeof(float)) {
        float* p = arr.data<float>();
        w.assign(p, p + arr.num_vals);
    } else if (arr.word_size == sizeof(double)) {
        double* p = arr.data<double>();
        w.resize(arr.num_vals);
        for (size_t i=0;i<arr.num_vals;++i) w[i] = static_cast<float>(p[i]);
    } else {
        throw std::runtime_error("Tipo pesi non supportato");
    }
    // (opzionale) normalizza a somma 1 per avere media pesata
    double s=0.0; for (float v: w) s += v;
    if (s > 0) for (auto& v: w) v = static_cast<float>(v / s);
    return w;
}

static void save_scores_npy(const std::string& path, const std::vector<float>& scores) {
    std::vector<size_t> shape = { scores.size() };
    cnpy::npy_save(path, scores.data(), shape, "w");
    std::cout << "Salvato: " << path << std::endl;
}

// Carica batch N×P (float32) se esiste, altrimenti legge directory con image_*.npy (uint8 o float)
static std::vector<float> load_images_batch_or_dir(const std::string& batch_path,
                                                   const std::string& images_dir,
                                                   size_t& N_out) {
    if (!batch_path.empty() && fs::exists(batch_path)) {
        std::cout << "Carico batch immagini: " << batch_path << std::endl;
        cnpy::NpyArray arr = cnpy::npy_load(batch_path);
        if (arr.shape.size() != 2 || arr.shape[1] != P)
            throw std::runtime_error("images_batch.npy deve essere N x 50176 float32");
        if (arr.word_size != sizeof(float))
            throw std::runtime_error("images_batch.npy deve essere float32");
        N_out = arr.shape[0];
        const float* p = arr.data<float>();
        return std::vector<float>(p, p + N_out*P);
    }

    // fallback: directory
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(images_dir)) {
        if (e.path().extension() == ".npy" &&
            e.path().filename().string().rfind("image_", 0) == 0) {
            files.push_back(e.path());
        }
    }
    if (files.empty()) throw std::runtime_error("Nessuna immagine trovata in " + images_dir);
    std::sort(files.begin(), files.end());

    std::vector<float> all;
    all.reserve(files.size()*P);

    for (auto& f : files) {
        cnpy::NpyArray a = cnpy::npy_load(f.string());
        // supporta (224,224) uint8/float o array flatten P
        if (a.shape.size() == 2 && a.shape[0]==H && a.shape[1]==W) {
            if (a.word_size == sizeof(uint8_t)) {
                const uint8_t* p = a.data<uint8_t>();
                for (int i=0;i<P;++i) all.push_back(p[i] / 255.0f); // [0,1]
            } else if (a.word_size == sizeof(float)) {
                const float* p = a.data<float>();
                all.insert(all.end(), p, p+P);
            } else {
                throw std::runtime_error("Formato immagine non supportato (224x224)");
            }
        } else if (a.shape.size()==1 && a.shape[0]==P) {
            if (a.word_size != sizeof(float))
                throw std::runtime_error("Immagine flatten deve essere float32");
            const float* p = a.data<float>();
            all.insert(all.end(), p, p+P);
        } else {
            throw std::runtime_error("Dimensione immagine inattesa in " + f.string());
        }
    }
    N_out = files.size();
    std::cout << "Caricate " << N_out << " immagini da directory.\n";
    return all;
}

// (opzionale) sequenziale CPU per confronto tempi/correttezza
static std::vector<float> cpu_dot_scores(const std::vector<float>& imgs,
                                         const std::vector<float>& w,
                                         size_t N) {
    std::vector<float> s(N, 0.0f);
    for (size_t n=0;n<N;++n) {
        const float* img = imgs.data() + n*P;
        double acc=0.0;
        for (int i=0;i<P;++i) acc += img[i]*w[i];
        s[n] = static_cast<float>(acc);
    }
    return s;
}

// -------- MAIN --------
int main(int argc, char** argv) {
    try {
        // Path di default
        std::string weights_path = "../data/weights/static_weights_224x224.npy";
        std::string images_batch = "../data/images_batch_224x224.npy"; // se non esiste, usa directory
        std::string images_dir   = "../data/ChestMNIST_Images";
        std::string out_scores   = "../results/scores_gpu.npy";
        bool run_cpu_baseline = true;
        int block_size = 256;

        // parse argomenti semplici
        for (int i=1;i<argc;++i) {
            if (!std::strcmp(argv[i], "--weights") && i+1<argc) weights_path = argv[++i];
            else if (!std::strcmp(argv[i], "--batch") && i+1<argc) images_batch = argv[++i];
            else if (!std::strcmp(argv[i], "--images") && i+1<argc) images_dir = argv[++i];
            else if (!std::strcmp(argv[i], "--out") && i+1<argc) out_scores = argv[++i];
            else if (!std::strcmp(argv[i], "--bs") && i+1<argc) block_size = std::stoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--no-cpu")) run_cpu_baseline = false;
        }

        // crea cartella results
        try { fs::create_directories(fs::path(out_scores).parent_path()); } catch(...) {}

        // Carica pesi (normalizzati a somma = 1)
        auto weights = load_weights_or_die(weights_path);

        // Carica immagini (batch o dir)
        size_t N = 0;
        auto images = load_images_batch_or_dir(images_batch, images_dir, N);
        if (images.size() != N*P) throw std::runtime_error("Dimensioni immagini non coerenti");

        std::cout << "N = " << N << ", P = " << P << ", block_size = " << block_size << std::endl;

        // --- CPU baseline (opzionale) ---
        std::vector<float> cpu_scores;
        double cpu_ms = 0.0;
        if (run_cpu_baseline) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cpu_scores = cpu_dot_scores(images, weights, N);
            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "[CPU] dot total: " << cpu_ms << " ms" << std::endl;
        }

        // --- GPU alloc ---
        float *d_images=nullptr, *d_weights=nullptr, *d_scores=nullptr;
        CHECK_CUDA(cudaMalloc((void**)&d_images,  N*P*sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_weights, P*sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_scores,  N*sizeof(float)));

        // copy H2D
        CHECK_CUDA(cudaMemcpy(d_images,  images.data(),  N*P*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_weights, weights.data(), P*sizeof(float),   cudaMemcpyHostToDevice));

        // --- GPU kernel ---
        dim3 grid(N);
        dim3 block(block_size);
        size_t shmem = block_size * sizeof(float);

        cudaEvent_t ev_start, ev_stop;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));

        CHECK_CUDA(cudaEventRecord(ev_start));
        dot_per_image_kernel<<<grid, block, shmem>>>(d_images, d_weights, d_scores, P);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        float gpu_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
        std::cout << "[GPU] kernel time: " << gpu_ms << " ms" << std::endl;

        // copy D2H
        std::vector<float> scores(N);
        CHECK_CUDA(cudaMemcpy(scores.data(), d_scores, N*sizeof(float), cudaMemcpyDeviceToHost));

        // confronta con CPU (se fatto)
        if (run_cpu_baseline) {
            double mae = 0.0, maxae = 0.0;
            for (size_t i=0;i<N;++i) {
                double ae = std::abs((double)scores[i] - (double)cpu_scores[i]);
                mae += ae;
                if (ae > maxae) maxae = ae;
            }
            mae /= N;
            std::cout << "Confronto CPU vs GPU -> MAE: " << mae << "  maxAE: " << maxae << std::endl;
        }

        // salva risultati GPU
        save_scores_npy(out_scores, scores);

        // cleanup
        CHECK_CUDA(cudaFree(d_images));
        CHECK_CUDA(cudaFree(d_weights));
        CHECK_CUDA(cudaFree(d_scores));
        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_stop));

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
        return 1;
    }
}
