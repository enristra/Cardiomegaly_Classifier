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
#define P (H*W)



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


//--------KERNEL

__global__ void dot_kernel(const float *images, const float *weights, float *scores, int pixels_per_img) {
    int img_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    const float *img = images + img_idx * pixels_per_img;

    float sum = 0.0f;
    for (int i = tid; i < pixels_per_img; i += blockDim.x) {
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

//----------------I/O------


float* load_weights(const char* filename) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    if (arr.shape.size() != 2 || arr.shape[0] != (size_t)H || arr.shape[1] != (size_t)W) {
        printf("Errore: dimensione pesi non valida (atteso 224x224)\n");
        exit(1);
    }
    if(arr.word_size != sizeof(float)){
        printf("Errore: tipo pesi non supportato (atteso float32)\n");
        exit(1);
    }


    if(!weights){
        printf("Errore malloc pesi\n");
        exit(1);
    }
    float *weights = (float*)malloc(P * sizeof(float));

    
    float* p = arr.data<float>();
    for (int i = 0; i < P; i++) weights[i] = p[i];
        
    
    return weights;
}

float* load_images(const char* filename, int *N) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    if (arr.shape.size() != 2 || arr.shape[1] != (size_t)P) {
        printf("Errore: dimensione immagini non valida\n");
        exit(1);
    }

    *N = (int)arr.shape[0];
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
    vector<size_t> shape = { (size_t)N };
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

    
    //caricamento dati host
    
    cudaError_t err;
size_t bytes_imgs= (size_t)N * P * sizeof(float);
size_t bytes_w    = (size_t)P * sizeof(float);
size_t bytes_out  = (size_t)N * sizeof(float);

float *d_images = NULL, *d_weights = NULL, *d_scores = NULL;

int N;
float *weights = load_weights(weights_path);
float *images = load_images(images_path, &N);


    
    

    //Allocazione memoria
    err =  cudaMalloc((void**)&d_images, bytes_imgs);
    if(err != cudaSuccess){
        printf("Errore allocazione d_images: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc ((void**)&d_weights, bytes_w);
    if (err != cudaSuccess){
        printf("Errore allocazione d_weights: %s\n");
        return 1;
    }

    err = cudaMalloc((void**)&d_scores, bytes_out);
    if(err != cudaSuccess){
        printf("Errore allocazione d_scores: %s\n");
        return 1;
    }

    printf("Memoria allocata con successo sulla GPU. \n");


    // Trasferimento dati

    err = cudaMemcpy(d_images, images, bytes_imgs, cudaMemcpyHostToDevice);

    if(err != cudaSuccess){
        printf("Errore cudaMemcpy H2D images: %s\n", cudaGetErrorString(err));
        return 1;
    }


    err = cudaMemcpy(d_weights, weights, bytes_w, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("Errore cudaMemcpy H2D weights: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int blockSize = 256;
    size_t shmem = (size_t)blockSize * sizeof(float);

    dot_kernel<<<N, blockSize, shmem >>>(d_images, d_weights, d_scores, P);

    err= cudaGetLastError();
    if (err != cudaSuccess){
        printf("Errore nel lancio del kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(scores, d_scores, bytes_out, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess){
        printf("Errore cudaMemcpy D2H scores: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err =cudaDeviceSynchronize();
    if (err != cudaSuccess){
        printf("Errore in cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        return 1;
    }

    save_scores(out_path, scores, N);

    err = cudaFree(d_images);
    if (err != cudaSuccess) printf("Warning cudaFree(d_images): %s\n", cudaGetErrorString(err));


    err = cudaFree(d_weights);
    if (err != cudaSuccess) printf("Warning cudaFree(d_weights): %s\n", cudaGetErrorString(err));

    err = cudaFree(d_scores);
    if (err != cudaSuccess) printf("Warning cudaFree(d_scores): %s\n", cudaGetErrorString(err));


    free(weights);
    free(images);
    free(scores);

    return 0;
}
