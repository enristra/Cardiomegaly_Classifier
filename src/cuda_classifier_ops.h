#ifndef CUDA_CLASSIFIER_OPS_H
#define CUDA_CLASSIFIER_OPS_H

#include <string> // Per std::string
#include <vector> // Per std::vector

// Macro per il controllo degli errori CUDA
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__         \
                      << " - " << cudaGetErrorString(err) << " (" << err << ")"  \
                      << std::endl;                                               \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

// Funzione per allocare i pesi sulla GPU e copiarli dal CPU.
// Restituisce un puntatore ai pesi sulla GPU e imposta la dimensione.
void initCudaWeights(const std::vector<float>& host_weights, float** dev_weights, int* num_weights);

// Funzione per liberare la memoria dei pesi sulla GPU.
void freeCudaWeights(float* dev_weights);

// Funzione per calcolare lo score di un'immagine sulla GPU.
// Prende i dati dell'immagine dalla CPU, li copia sulla GPU,
// esegue il calcolo con i pesi già sulla GPU, e restituisce lo score finale.
float calculateScoreOnGPU(const std::vector<float>& image_data_host, const float* dev_weights, int num_weights);

#ifdef __cplusplus
}
#endif

#endif // CUDA_CLASSIFIER_OPS_H