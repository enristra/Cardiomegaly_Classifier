#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/stat.h>

#include <string>
#include <vector>
#include <algorithm>
#include "cnpy.h"

using namespace std;

#define H 224
#define W 224
#define P (X*W)



//Utilities
static int file_exists(const char* path){
    FILE* f = fopen(path, "rb");
    if(!f) return 0;
    fclose(f);
    return 1;
}

static int dir_exists(const char* path){
    struct  stat st;
    return (stat(path, &st)==0) && S_ISDIR(st.st_mode);
}



__global__ void dot_kernel(const float *images, const float *weights, float *scores, int pixels_per_img) {
    int img_idx = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];

    const float *img = images + img_idx * P;

    float sum = 0.0f;
    for (int i = tid; i < P; i += blockDim.x) {
        sum += img[i] * weights[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        scores[img_idx] = sdata[0];
    }
}

float* load_weights(const char* filename) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    if (arr.shape.size() != 2 || arr.shape[0] != H || arr.shape[1] != W) {
        printf("Errore: dimensione pesi non valida\n");
        exit(1);
    }
    float *weights = (float*)malloc(P * sizeof(float));
    if (arr.word_size == sizeof(float)) {
        float* p = arr.data<float>();
        for (int i = 0; i < P; i++) weights[i] = p[i];
    } else {
        printf("Errore: tipo pesi non supportato\n");
        exit(1);
    }
    return weights;
}

float* load_images(const char* filename, int *N) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    if (arr.shape.size() != 2 || arr.shape[1] != P) {
        printf("Errore: dimensione immagini non valida\n");
        exit(1);
    }
    *N = arr.shape[0];
    float *images = (float*)malloc((*N) * P * sizeof(float));
    if (arr.word_size == sizeof(float)) {
        float* p = arr.data<float>();
        for (int i = 0; i < (*N) * P; i++) images[i] = p[i];
    } else {
        printf("Errore: tipo immagini non supportato\n");
        exit(1);
    }
    return images;
}

void save_scores(const char* filename, float *scores, int N) {
    std::vector<size_t> shape = { (size_t)N };
    cnpy::npy_save(filename, scores, shape, "w");
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Uso: %s <weights.npy> <images.npy> <out.npy>\n", argv[0]);
        return 1;
    }

    const char* weights_path = argv[1];
    const char* images_path = argv[2];
    const char* out_path = argv[3];

    int N;
    float *weights = load_weights(weights_path);
    float *images = load_images(images_path, &N);
    float *scores = (float*)malloc(N * sizeof(float));

    float *d_images, *d_weights, *d_scores;
    cudaMalloc((void**)&d_images, N * P * sizeof(float));
    cudaMalloc((void**)&d_weights, P * sizeof(float));
    cudaMalloc((void**)&d_scores, N * sizeof(float));

    cudaMemcpy(d_images, images, N * P * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, P * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    dot_kernel<<<N, blockSize, blockSize * sizeof(float)>>>(d_images, d_weights, d_scores, N);

    cudaMemcpy(scores, d_scores, N * sizeof(float), cudaMemcpyDeviceToHost);

    save_scores(out_path, scores, N);

    cudaFree(d_images);
    cudaFree(d_weights);
    cudaFree(d_scores);
    free(weights);
    free(images);
    free(scores);

    return 0;
}
