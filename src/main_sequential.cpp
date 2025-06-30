#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include "cnpy.h"

namespace fs = std::filesystem;

struct ClassificationResult {
    std::string filename;
    float score;
    bool predicted;
    bool actual;
    bool correct;
};

struct Metrics {
    int true_positives = 0;
    int true_negatives = 0;
    int false_positives = 0;
    int false_negatives = 0;
    
    float accuracy() const {
        int total = true_positives + true_negatives + false_positives + false_negatives;
        return total > 0 ? float(true_positives + true_negatives) / total : 0.0f;
    }
    
    float sensitivity() const {
        int actual_positives = true_positives + false_negatives;
        return actual_positives > 0 ? float(true_positives) / actual_positives : 0.0f;
    }
    
    float specificity() const {
        int actual_negatives = true_negatives + false_positives;
        return actual_negatives > 0 ? float(true_negatives) / actual_negatives : 0.0f;
    }
    
    float precision() const {
        int predicted_positives = true_positives + false_positives;
        return predicted_positives > 0 ? float(true_positives) / predicted_positives : 0.0f;
    }
};

class CardiomegalyClassifier {
private:
    std::vector<float> weights;
    std::vector<int> labels;
    float threshold;
    
public:
    CardiomegalyClassifier(const std::string& weights_file, const std::string& labels_file, 
                          float classification_threshold = 0.5f) 
        : threshold(classification_threshold) {
        loadWeights(weights_file);
        loadLabels(labels_file);
    }
    
    void loadWeights(const std::string& filename) {
        std::cout << "Caricamento pesi da: " << filename << std::endl;
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        
        if (arr.shape.size() != 2 || arr.shape[0] != 224 || arr.shape[1] != 224) {
            throw std::runtime_error("I pesi devono essere 224x224");
        }
        
        // Converti a float se necessario
        if (arr.word_size == sizeof(float)) {
            float* data = arr.data<float>();
            weights.assign(data, data + arr.num_vals);
        } else if (arr.word_size == sizeof(double)) {
            double* data = arr.data<double>();
            weights.reserve(arr.num_vals);
            for (size_t i = 0; i < arr.num_vals; ++i) {
                weights.push_back(static_cast<float>(data[i]));
            }
        } else {
            throw std::runtime_error("Tipo di peso non supportato");
        }
        
        std::cout << "Pesi caricati: " << weights.size() << " elementi" << std::endl;
    }
    
    void loadLabels(const std::string& filename) {
        std::cout << "Caricamento labels da: " << filename << std::endl;
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        
        // Verifica che le labels siano un array 2D (num_samples, num_classes)
        // e che il numero di classi sia 14 (come ci aspettiamo da ChestMNIST)
        if (arr.shape.size() != 2 || arr.shape[1] != 14) {
            throw std::runtime_error("Le labels devono essere in formato (N, 14) come da dataset ChestMNIST originale.");
        }

        size_t num_samples = arr.shape[0];
        size_t label_index_cardiomegaly = 1; // Cardiomegalia è all'indice 1 (0-based)

        labels.clear();
        labels.reserve(num_samples); // Prealloca spazio per N etichette (una per immagine)

        // Carica le labels in base al loro tipo di dato
        // Accedi all'elemento corretto: (riga * num_colonne) + indice_colonna
        if (arr.word_size == sizeof(int32_t)) {
            int32_t* data = arr.data<int32_t>();
            for (size_t i = 0; i < num_samples; ++i) {
                labels.push_back(static_cast<int>(data[i * arr.shape[1] + label_index_cardiomegaly]));
            }
        } else if (arr.word_size == sizeof(int64_t)) { // Supporta anche int64_t se il file è così
            int64_t* data = arr.data<int64_t>();
            for (size_t i = 0; i < num_samples; ++i) {
                labels.push_back(static_cast<int>(data[i * arr.shape[1] + label_index_cardiomegaly]));
            }
        } else if (arr.word_size == sizeof(float)) { // Supporta anche float, per flessibilità
            float* data = arr.data<float>();
            for (size_t i = 0; i < num_samples; ++i) {
                labels.push_back(static_cast<int>(data[i * arr.shape[1] + label_index_cardiomegaly]));
            }
        } else {
            throw std::runtime_error("Tipo di label non supportato in labels.npy.");
        }
        
        std::cout << "Labels caricate: " << labels.size() << " elementi (solo cardiomegalia)" << std::endl;
    }
    
    float calculateScore(const std::string& image_file) {
        cnpy::NpyArray arr = cnpy::npy_load(image_file);
        
        // Converti l'immagine a float normalizzato
        std::vector<float> image_data;
        
        if (arr.word_size == sizeof(uint8_t)) {
            // Immagini uint8 (0-255) -> normalizza a (0-1)
            uint8_t* data = arr.data<uint8_t>();
            image_data.reserve(arr.num_vals);
            for (size_t i = 0; i < arr.num_vals; ++i) {
                float normalized_0_1 = static_cast<float>(data[i]) / 255.0f;     // Scala a 0-1
                float normalized_neg1_1 = (normalized_0_1 - 0.5f) / 0.5f;       // Scala a -1 a 1 (media=0.5, std=0.5)
                image_data.push_back(normalized_neg1_1);
            }
        } else if (arr.word_size == sizeof(float)) {
            float* data = arr.data<float>();
            image_data.assign(data, data + arr.num_vals);
        } else {
            throw std::runtime_error("Tipo di immagine non supportato");
        }
        
        // Determina il numero di canali
        size_t pixels_per_channel = 224 * 224;
        size_t num_channels = image_data.size() / pixels_per_channel;
        
        if (image_data.size() % pixels_per_channel != 0) {
            throw std::runtime_error("Dimensioni immagine non valide");
        }
        
        // Calcola lo score applicando i pesi
        float total_score = 0.0f;
        
        for (size_t c = 0; c < num_channels; ++c) {
            for (size_t i = 0; i < pixels_per_channel; ++i) {
                size_t img_idx = c * pixels_per_channel + i;
                total_score += image_data[img_idx] * weights[i];
            }
        }
        
        // Normalizza per il numero di canali se necessario
        return total_score / static_cast<float>(num_channels);
    }
    
    std::vector<ClassificationResult> classifyAll(const std::string& images_dir) {
        std::vector<ClassificationResult> results;
        
        // Ottieni tutti i file immagine ordinati
        std::vector<fs::path> image_files;
        for (const auto& entry : fs::directory_iterator(images_dir)) {
            if (entry.path().extension() == ".npy" && 
                entry.path().filename().string().find("image_") == 0) {
                image_files.push_back(entry.path());
            }
        }
        
        std::sort(image_files.begin(), image_files.end());
        
        std::cout << "Classificazione di " << image_files.size() << " immagini..." << std::endl;
        
        for (size_t i = 0; i < image_files.size() && i < labels.size(); ++i) {
            try {
                ClassificationResult result;
                result.filename = image_files[i].filename().string();
                result.score = calculateScore(image_files[i].string());
                result.predicted = result.score > threshold;
                result.actual = labels[i] > 0;  // Assumendo 0=sano, >0=cardiomegalia
                result.correct = (result.predicted == result.actual);
                
                results.push_back(result);
                
                if ((i + 1) % 20 == 0) {
                    std::cout << "Elaborate " << (i + 1) << " immagini..." << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Errore nell'elaborazione di " << image_files[i].filename() 
                         << ": " << e.what() << std::endl;
            }
        }
        
        return results;
    }
    
    Metrics calculateMetrics(const std::vector<ClassificationResult>& results) {
        Metrics metrics;
        
        for (const auto& result : results) {
            if (result.predicted && result.actual) {
                metrics.true_positives++;
            } else if (result.predicted && !result.actual) {
                metrics.false_positives++;
            } else if (!result.predicted && result.actual) {
                metrics.false_negatives++;
            } else {
                metrics.true_negatives++;
            }
        }
        
        return metrics;
    }
    
    void setThreshold(float new_threshold) {
        threshold = new_threshold;
    }
    
    float getThreshold() const {
        return threshold;
    }
};

void printResults(const std::vector<ClassificationResult>& results, const Metrics& metrics) {
    std::cout << "\n=== RISULTATI CLASSIFICAZIONE ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    std::cout << "\nMETRICHE:" << std::endl;
    std::cout << "Accuracy:    " << metrics.accuracy() * 100 << "%" << std::endl;
    std::cout << "Sensitivity: " << metrics.sensitivity() * 100 << "%" << std::endl;
    std::cout << "Specificity: " << metrics.specificity() * 100 << "%" << std::endl;
    std::cout << "Precision:   " << metrics.precision() * 100 << "%" << std::endl;
    
    std::cout << "\nCONFUSION MATRIX:" << std::endl;
    std::cout << "                 Predicted" << std::endl;
    std::cout << "                 Neg   Pos" << std::endl;
    std::cout << "Actual    Neg   " << std::setw(4) << metrics.true_negatives 
              << "  " << std::setw(4) << metrics.false_positives << std::endl;
    std::cout << "          Pos   " << std::setw(4) << metrics.false_negatives 
              << "  " << std::setw(4) << metrics.true_positives << std::endl;
    
    std::cout << "\nPRIMI 10 RISULTATI:" << std::endl;
    std::cout << "File                Score    Pred  Actual  Correct" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    
    for (size_t i = 0; i < std::min(size_t(10), results.size()); ++i) {
        const auto& r = results[i];
        std::cout << std::setw(15) << r.filename.substr(0, 15) << "  "
                  << std::setw(7) << r.score << "  "
                  << std::setw(4) << (r.predicted ? "POS" : "NEG") << "  "
                  << std::setw(6) << (r.actual ? "POS" : "NEG") << "  "
                  << (r.correct ? "✓" : "✗") << std::endl;
    }
}

int main() {
    try {
        // Parametri
        std::string weights_file = "../data/weights/cardiomegaly_weights_224x224_trained.npy";
        std::string labels_file = "../data/ChestMNIST_Images/labels.npy";
        std::string images_dir = "../data/ChestMNIST_Images";
        
        float threshold = 0.0f;  // Threshold per classificazione
        
        // Crea il classificatore
        CardiomegalyClassifier classifier(weights_file, labels_file, threshold);
        
        // Classifica tutte le immagini
        auto results = classifier.classifyAll(images_dir);
        
        // Calcola metriche
        auto metrics = classifier.calculateMetrics(results);
        
        // Stampa risultati
        printResults(results, metrics);
        
        // Opzionale: test con threshold diversi
        std::cout << "\n=== TEST THRESHOLD DIVERSI ===" << std::endl;
        std::vector<float> thresholds_to_test = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}; // Suggerimento: testare intorno a 0
        
        for (float th : thresholds_to_test) {
            classifier.setThreshold(th);
            // Riclassifica con nuovo threshold
            for (auto& result : results) {
                result.predicted = result.score > th;
                result.correct = (result.predicted == result.actual);
            }
            
            auto th_metrics = classifier.calculateMetrics(results);
            std::cout << "Threshold " << th << ": Acc=" 
                      << std::setprecision(3) << th_metrics.accuracy() * 100 
                      << "%, Sens=" << th_metrics.sensitivity() * 100 
                      << "%, Spec=" << th_metrics.specificity() * 100 << "%" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
        return 1;
    }
}