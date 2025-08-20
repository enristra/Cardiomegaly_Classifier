#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "cnpy.h"


using namespace std;

#define H 224
#define W 224
#define P (H*W)

#define BLOCK_SIZE 256

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
    const int img = blockDim.x;
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


static float* load_weights(const char* filename) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    // accetto shape (H,W) o (P,)
    if (!((arr.shape.size() == 2 && arr.shape[0] == (size_t)H && arr.shape[1] == (size_t)W) ||
          (arr.shape.size() == 1 && arr.shape[0] == (size_t)P))) {
        fprintf(stderr, "Errore: dimensione pesi non valida (atteso 224x224 o 50176)\n");
        exit(1);
    }
   if (arr.word_size != sizeof(float)) {
        fprintf(stderr, "Errore: tipo pesi non supportato (atteso float32)\n");
        exit(1);
    }

    float* weights = (float*)malloc(P * sizeof(float));
    if (!weights) {
        fprintf(stderr, "Errore malloc pesi\n");
        exit(1);
    }
    
    const float* p = arr.data<float>();

    // Se (H,W), i dati sono già contigui per riga: appiattisco
    for (int i = 0; i < P; ++i) weights[i] = p[i];    
    
    return weights;
}

static float* load_images(const char* filename, int* N) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    
    // Accetto (N, P) oppure (N, H, W) oppure (N, 1, H, W)
    bool ok = false;
    if (arr.shape.size() == 2 && arr.shape[1] == (size_t)P) ok = true;
    if (arr.shape.size() == 3 && arr.shape[1] == (size_t)H && arr.shape[2] == (size_t)W) ok = true;
    if (arr.shape.size() == 4 && arr.shape[1] == 1 && arr.shape[2] == (size_t)H && arr.shape[3] == (size_t)W) ok = true;

    if (!ok) {
        fprintf(stderr, "Errore: dimensione immagini non valida (atteso (N,P) o (N,H,W) o (N,1,H,W))\n");
        exit(1);
    }
    if (arr.word_size != sizeof(float)) {
        fprintf(stderr, "Errore: tipo immagini non supportato (atteso float32)\n");
        exit(1);
    }

    int n_imgs = (int)arr.shape[0];
    *N = n_imgs;

    float* images = (float*)malloc((size_t)n_imgs * P * sizeof(float));
    if (!images) {
        fprintf(stderr, "Errore malloc immagini\n");
        exit(1);
    }
   const float* src = arr.data<float>();

    if (arr.shape.size() == 2) {
        // (N,P) già appiattito
        memcpy(images, src, (size_t)n_imgs * P * sizeof(float));
    } else if (arr.shape.size() == 3) {
        // (N,H,W) -> (N,P)
        for (int n = 0; n < n_imgs; ++n) {
            memcpy(images + (size_t)n * P, src + (size_t)n * P, (size_t)P * sizeof(float));
        }
    } else {
        // (N,1,H,W) -> (N,P)
        for (int n = 0; n < n_imgs; ++n) {
            const float* in = src + (size_t)n * (size_t)(1 * P);
            memcpy(images + (size_t)n * P, in, (size_t)P * sizeof(float));
        }
    }

    return images;
}

static void save_scores(const char* filename, float *scores, int N) {
    vector<size_t> shape = { (size_t)N };
    cnpy::npy_save(filename, scores, shape, "w");
}


// CLI
struct Args{
    string weights_path;
    string images_path;
    string out_path;
    string kernel = "naive"; //naive || reduce
    int block = BLOCK_SIZE;
};

static void print_usage(const char* prog){
    fprintf(stderr,
        "Uso:\n"
        "   %s <weights.npy> <images.npy> <out.npy> [--kernel naive | reduce ] [--block N]\n"
        "default: --kernel naive, -- block %d\n",
        prog, BLOCK_SIZE
    );
}

static Args parse_args(int argc, char** argv){
    Args a;
    if (argc < 4){
        print_usage(argv[0]);
        exit(1);
    }
    a.weights_path = argv[1];
    a.images_path = argv[2];
    a.out_path = argv[3];
 
    for (int i = 4; i < argc; ++i) {
        string k = argv[i];
        if (k == "--kernel" && i + 1 < argc) { a.kernel = argv[++i]; }
        else if (k == "--block" && i + 1 < argc) { a.block = atoi(argv[++i]); }
        else {
            fprintf(stderr, "Argomento sconosciuto o incompleto: %s\n", k.c_str());
            print_usage(argv[0]);
            exit(1);
        }
    }

    if (a.kernel != "naive" && a.kernel != "reduce") {
        fprintf(stderr, "Valore non valido per --kernel (usa naive|reduce)\n");
        exit(1);
    }
    if (a.block <= 0 || a.block > 1024) {
        fprintf(stderr, "Valore non valido per --block (1..1024)\n");
        exit(1);
    }
    return a;

}

int main(int argc, char** argv) {
   Args args = parse_args(argc, argv);

   //caricamento dati su host
   int N = 0;
   float* weights = load_weights(args.weights_path.c_str());
   float* images = load_images(args.images_path.c_str(), &N);

   //allocazioni in base a N

    size_t bytes_imgs= (size_t)N * P * sizeof(float);
    size_t bytes_w    = (size_t)P * sizeof(float);
    size_t bytes_out  = (size_t)N * sizeof(float);
    
    float *d_images = nullptr, *d_weights = nullptr, *d_scores = nullptr;
    float *scores = (float*)malloc(bytes_out);
    if(!scores){
        fprintf(stderr, "Errore malloc scores\n");
        return 1;
    }

    CUDA_CHECK(cudaMalloc((void**)&d_images,  bytes_imgs));
    CUDA_CHECK(cudaMalloc((void**)&d_weights, bytes_w));
    CUDA_CHECK(cudaMalloc((void**)&d_scores,  bytes_out));

    CUDA_CHECK(cudaMemcpy(d_images,  images,  bytes_imgs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, bytes_w,   cudaMemcpyHostToDevice));

    // Se  kernel naive azzero l'output
    CUDA_CHECK(cudaMemset(d_scores, 0, bytes_out));

    const int blockSize = args.block;
    const dim3 grid(N);
    const dim3 block(blockSize);
    const size_t shmem = (size_t)blockSize * sizeof(float);

    if (args.kernel == "naive") {
        dot_kernel_naive_atomic<<<grid, block>>>(d_images, d_weights, d_scores, P);
    } else { // reduce
        dot_kernel_reduce_shared<<<grid, block, shmem>>>(d_images, d_weights, d_scores, P);
    }
   
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(scores, d_scores, bytes_out, cudaMemcpyDeviceToHost));

    save_scores(args.out_path.c_str(), scores, N);

   
    //Deallocazione

    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_scores));
    free(weights);
    free(images);
    free(scores);

    printf("OK: kernel=%s, block=%d, N=%d → salvato %s\n",
           args.kernel.c_str(), blockSize, N, args.out_path.c_str());
    return 0;
}
