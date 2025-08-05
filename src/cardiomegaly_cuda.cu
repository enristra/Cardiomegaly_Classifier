#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include "cnpy.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Kernel CUDA per calcolare il punteggio per ogni immagine
__global__ void calculateScoresKernel(const float* images, const float* weights, float* scores, int width, int height, int num_images) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx = blockIdx.z;

    if (x < width && y < height && img_idx < num_images) {
        int pixel_index = img_idx * width * height + y * width + x;
        atomicAdd(&scores[img_idx], images[pixel_index] * weights[y * width + x]);
    }
}

// Calcolo sequenziale CPU
float cpuSequential(const std::vector<float>& images, const std::vector<float>& weights, int width, int height, int num_images, std::vector<float>& scores) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int img = 0; img < num_images; img++) {
        float score = 0.0f;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = img * width * height + y * width + x;
                score += images[idx] * weights[y * width + x];
            }
        }
        scores[img] = score;
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

int main() {
    try {
        std::string weights_file = "data/weights/cardiomegaly_weights_224x224_trained_CROP.npy";
        std::string images_file = "data/images/test_images.npy";
        int width = 224, height = 224;

        // Carica pesi
        cnpy::NpyArray weights_arr = cnpy::npy_load(weights_file);
        const float* weights_data = weights_arr.data<float>();
        std::vector<float> weights(weights_data, weights_data + weights_arr.num_vals);

        // Carica immagini
        cnpy::NpyArray images_arr = cnpy::npy_load(images_file);
        const float* images_data = images_arr.data<float>();
        int num_images = images_arr.shape[0];
        std::vector<float> images(images_data, images_data + images_arr.num_vals);

        std::cout << "Numero immagini: " << num_images << ", dimensioni: " << width << "x" << height << std::endl;

        // Limita il numero di immagini per evitare problemi di VRAM
        int max_images = 1000;  // puoi modificare in base alla GPU
        if (num_images > max_images) {
            num_images = max_images;
            images.resize(num_images * width * height);
            std::cout << "Eseguo su subset di " << num_images << " immagini.\n";
        }

        // Vettori per risultati
        std::vector<float> scores_cpu(num_images, 0.0f);
        std::vector<float> scores_gpu(num_images, 0.0f);

        // --- CPU Baseline ---
        float cpu_time = cpuSequential(images, weights, width, height, num_images, scores_cpu);
        std::cout << "Tempo CPU: " << cpu_time << " ms\n";

        // --- Allocazione GPU ---
        float *d_images, *d_weights, *d_scores;
        size_t img_size = images.size() * sizeof(float);
        size_t weights_size = weights.size() * sizeof(float);
        size_t scores_size = scores_gpu.size() * sizeof(float);

        cudaMalloc(&d_images, img_size);
        cudaMalloc(&d_weights, weights_size);
        cudaMalloc(&d_scores, scores_size);

        cudaMemcpy(d_images, images.data(), img_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, weights.data(), weights_size, cudaMemcpyHostToDevice);
        cudaMemset(d_scores, 0, scores_size);

        // --- Configurazione kernel ---
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  num_images);

        // --- Timing GPU ---
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        calculateScoresKernel<<<grid, block>>>(d_images, d_weights, d_scores, width, height, num_images);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);

        cudaMemcpy(scores_gpu.data(), d_scores, scores_size, cudaMemcpyDeviceToHost);

        std::cout << "Tempo GPU: " << gpu_time << " ms\n";

        cudaFree(d_images);
        cudaFree(d_weights);
        cudaFree(d_scores);

        // --- Salvataggio risultati ---
        std::ofstream outfile("results.csv");
        outfile << "image_id,cpu_score,gpu_score\n";
        for (int i = 0; i < num_images; i++) {
            outfile << i << "," << scores_cpu[i] << "," << scores_gpu[i] << "\n";
        }
        outfile.close();
        std::cout << "Risultati salvati in results.csv\n";

    } catch (std::exception& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
    }

    return 0;
}
