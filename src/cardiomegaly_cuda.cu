#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include "cnpy.h"

#define IMG_SIZE 224
#define PIXELS (IMG_SIZE * IMG_SIZE)

// Kernel 1: moltiplicazione pixel * peso
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

// Funzione host con misurazione tempo GPU
float calculateScoreOnGPU(const std::vector<float>& image, const std::vector<float>& weights, float& gpuTimeMs) {
    int size = PIXELS;
    size_t bytes = size * sizeof(float);

    float *d_image, *d_weights, *d_partial;
    cudaMalloc(&d_image, bytes);
    cudaMalloc(&d_weights, bytes);
    cudaMalloc(&d_partial, bytes);

    cudaMemcpy(d_image, image.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), bytes, cudaMemcpyHostToDevice);

    // Configurazione kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Eventi CUDA per timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 1. Kernel di moltiplicazione
    multiplyKernel<<<blocks, threads>>>(d_image, d_weights, d_partial, size);

    // 2. Riduzione
    int currentSize = size;
    while (currentSize > 1) {
        int smemSize = threads * sizeof(float);
        blocks = (currentSize + threads - 1) / threads;
        reduceSumKernel<<<blocks, threads, smemSize>>>(d_partial, currentSize);
        currentSize = blocks;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop); // Tempo in millisecondi

    // Copia risultato finale
    float result;
    cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_weights);
    cudaFree(d_partial);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

int main() {
    try {
        std::string weights_file = "/scratch.hpc/enrico.strangio/Cardiomegaly_Classifier/data/weights/cardiomegaly_weights_224x224_trained_CROP.npy";
        std::string image_file = "/scratch.hpc/enrico.strangio/Cardiomegaly_Classifier/data/images/test_image.npy";

        // Carica pesi
        cnpy::NpyArray arr_w = cnpy::npy_load(weights_file);
        float* w_data = arr_w.data<float>();
        std::vector<float> weights(w_data, w_data + PIXELS);

        // Carica immagine
        cnpy::NpyArray arr_img = cnpy::npy_load(image_file);
        float* img_data = arr_img.data<float>();
        std::vector<float> image(img_data, img_data + PIXELS);

        // Misura tempo GPU
        float gpuTimeMs = 0.0f;
        float score = calculateScoreOnGPU(image, weights, gpuTimeMs);

        std::cout << "Score GPU: " << score << std::endl;
        std::cout << "Tempo GPU (moltiplicazione + riduzione): " << gpuTimeMs << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
