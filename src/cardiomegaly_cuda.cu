#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "cnpy.h"

#define IMG_SIZE 224
#define PIXELS (IMG_SIZE * IMG_SIZE)

// Kernel 1: calcolo pixel * peso
__global__ void multiplyKernel(const float* image, const float* weights, float* partialResults, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        partialResults[idx] = image[idx] * weights[idx];
    }
}

// Kernel 2: riduzione parallela
__global__ void reduceSumKernel(float* partialResults, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Carica in shared memory
    sdata[tid] = (idx < size) ? partialResults[idx] : 0.0f;
    __syncthreads();

    // Riduzione
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Primo thread del blocco scrive il risultato
    if (tid == 0) {
        partialResults[blockIdx.x] = sdata[0];
    }
}

// Funzione host per inferenza su una sola immagine
float calculateScoreOnGPU(const std::vector<float>& image, const std::vector<float>& weights) {
    int size = PIXELS;
    size_t bytes = size * sizeof(float);

    float *d_image, *d_weights, *d_partial;
    cudaMalloc(&d_image, bytes);
    cudaMalloc(&d_weights, bytes);
    cudaMalloc(&d_partial, bytes);

    cudaMemcpy(d_image, image.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), bytes, cudaMemcpyHostToDevice);

    // 1. Kernel di moltiplicazione
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    multiplyKernel<<<blocks, threads>>>(d_image, d_weights, d_partial, size);

    // 2. Riduzione
    int currentSize = size;
    while (currentSize > 1) {
        int smemSize = thr
