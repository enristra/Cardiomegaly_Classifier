#include "cuda_classifier_ops.h" // Includi l'header che dichiara le funzioni

#include <iostream> // Per std::cerr

// **************************** KERNEL CUDA ****************************
// Kernel per calcolare il prodotto elemento per elemento
__global__ void calculatePartialScoreKernel(const float* dev_imageData, const float* dev_weights, float* dev_partialSums, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        dev_partialSums[idx] = dev_imageData[idx] * dev_weights[idx];
    }
}

// Kernel per la riduzione parallela (somma dei risultati parziali)
__global__ void reduceSumKernel(float* dev_partialSums, int num_elements, float* dev_finalSum) {
    extern __shared__ float sdata[]; 

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid; // Per accedere agli elementi in modo coalescente

    sdata[tid] = 0; 
    if (i < num_elements) {
        sdata[tid] += dev_partialSums[i];
    }
    if (i + blockDim.x < num_elements) { 
        sdata[tid] += dev_partialSums[i + blockDim.x];
    }
    __syncthreads(); 

    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(dev_finalSum, sdata[0]); 
    }
}
// ********************************************************************************


// Implementazione delle funzioni dichiarate nell'header
void initCudaWeights(const std::vector<float>& host_weights, float** dev_weights, int* num_weights_ptr) {
    *num_weights_ptr = host_weights.size();
    CUDA_CHECK(cudaMalloc((void**)dev_weights, host_weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*dev_weights, host_weights.data(), host_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void freeCudaWeights(float* dev_weights) {
    if (dev_weights) {
        CUDA_CHECK(cudaFree(dev_weights));
    }
}

float calculateScoreOnGPU(const std::vector<float>& image_data_host, const float* dev_weights, int num_weights) {
    int num_pixels = image_data_host.size();
    
    // Verifica che la dimensione dell'immagine corrisponda alla dimensione dei pesi
    if (num_pixels != num_weights) {
        std::cerr << "Errore: Dimensione dell'immagine (" << num_pixels << ") non corrisponde alla dimensione dei pesi (" << num_weights << ")." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocazione memoria su Device (GPU) per i dati dell'immagine e i risultati intermedi
    float* dev_imageData;
    float* dev_partialSums;
    float* dev_finalScore;
    
    CUDA_CHECK(cudaMalloc((void**)&dev_imageData, num_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dev_partialSums, num_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dev_finalScore, sizeof(float)));
    
    // Inizializza il risultato finale a zero sulla GPU
    CUDA_CHECK(cudaMemset(dev_finalScore, 0, sizeof(float)));

    // Copia i dati dell'immagine da Host a Device
    CUDA_CHECK(cudaMemcpy(dev_imageData, image_data_host.data(), num_pixels * sizeof(float), cudaMemcpyHostToDevice));

    // Configurazione del lancio del Kernel per i prodotti parziali
    int threadsPerBlock = 256; 
    int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock; 

    // Lancia il Kernel per i prodotti parziali
    calculatePartialScoreKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_imageData, dev_weights, dev_partialSums, num_pixels);
    CUDA_CHECK(cudaDeviceSynchronize()); // Sincronizza per catturare errori

    // Lancia il Kernel per la riduzione finale
    // Per la riduzione, usiamo un solo blocco e una shared memory della dimensione di threadsPerBlock
    // Si noti che la dimensione della shared memory è passata come terzo argomento a <<< >>>
    reduceSumKernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(dev_partialSums, num_pixels, dev_finalScore);
    CUDA_CHECK(cudaDeviceSynchronize()); 

    // Copia il risultato finale da Device a Host
    float final_score_host;
    CUDA_CHECK(cudaMemcpy(&final_score_host, dev_finalScore, sizeof(float), cudaMemcpyDeviceToHost));

    // Liberare la memoria GPU allocata per questa immagine
    CUDA_CHECK(cudaFree(dev_imageData));
    CUDA_CHECK(cudaFree(dev_partialSums));
    CUDA_CHECK(cudaFree(dev_finalScore));
    
    return final_score_host;
}