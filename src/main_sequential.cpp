#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <limits>
#include "cnpy.h"
#include <fstream>
#include <cstring>

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
    std::vector<float> weights;   // 224*224
    std::vector<int> labels;      // N
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

        // Normalizza i pesi in modo che sommino a 1 (media pesata dei pixel)
        double s = 0.0;
        for (float w : weights) s += w;
        if (s > 0) {
            for (float& w : weights) w = static_cast<float>(w / s);
            std::cout << "Pesi normalizzati: somma = 1.0\n";
        } else {
            std::cout << "Attenzione: somma pesi = 0, nessuna normalizzazione applicata.\n";
        }
        
        std::cout << "Pesi caricati: " << weights.size() << " elementi" << std::endl;
    }
    
    void loadLabels(const std::string& filename) {
        std::cout << "Caricamento labels da: " << filename << std::endl;
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        
        if (arr.shape.size() != 2 || arr.shape[1] != 14) {
            throw std::runtime_error("Le labels devono essere (N,14) come ChestMNIST.");
        }

        size_t num_samples = arr.shape[0];
        size_t label_index_cardiomegaly = 1; // cardiomegalia

        labels.clear();
        labels.reserve(num_samples);

        if (arr.word_size == sizeof(int32_t)) {
            int32_t* data = arr.data<int32_t>();
            for (size_t i = 0; i < num_samples; ++i) {
                labels.push_back(static_cast<int>(data[i * arr.shape[1] + label_index_cardiomegaly]));
            }
        } else if (arr.word_size == sizeof(int64_t)) {
            int64_t* data = arr.data<int64_t>();
            for (size_t i = 0; i < num_samples; ++i) {
                labels.push_back(static_cast<int>(data[i * arr.shape[1] + label_index_cardiomegaly]));
            }
        } else if (arr.word_size == sizeof(float)) {
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
        
        std::vector<float> image_data;
        
        if (arr.word_size == sizeof(uint8_t)) {
            // Normalizza in [0,1] (NON più in [-1,1])
            uint8_t* data = arr.data<uint8_t>();
            image_data.reserve(arr.num_vals);
            for (size_t i = 0; i < arr.num_vals; ++i) {
                float normalized_0_1 = static_cast<float>(data[i]) / 255.0f;
                image_data.push_back(normalized_0_1);
            }
        } else if (arr.word_size == sizeof(float)) {
            float* data = arr.data<float>();
            image_data.assign(data, data + arr.num_vals);
        } else {
            throw std::runtime_error("Tipo di immagine non supportato");
        }
        
        size_t pixels_per_channel = 224 * 224;
        if (image_data.size() % pixels_per_channel != 0) {
            throw std::runtime_error("Dimensioni immagine non valide");
        }
        size_t num_channels = image_data.size() / pixels_per_channel;

        // Con pesi normalizzati, questo è la media pesata del canale.
        // Se più canali (RGB), media sui canali.
        float total_score = 0.0f;
        for (size_t c = 0; c < num_channels; ++c) {
            for (size_t i = 0; i < pixels_per_channel; ++i) {
                size_t img_idx = c * pixels_per_channel + i;
                total_score += image_data[img_idx] * weights[i];
            }
        }
        return total_score / static_cast<float>(num_channels);
    }
    
    std::vector<ClassificationResult> classifyAll(const std::string& images_dir) {
        std::vector<ClassificationResult> results;
        
        std::vector<fs::path> image_files;
        for (const auto& entry : fs::directory_iterator(images_dir)) {
            if (entry.path().extension() == ".npy" && 
                entry.path().filename().string().find("image_") == 0) {
                image_files.push_back(entry.path());
            }
        }
        std::sort(image_files.begin(), image_files.end());
        
        std::cout << "Classificazione di " << image_files.size() << " immagini..." << std::endl;
        
        for (size_t i = 0; i < 200 && i < labels.size(); ++i) {
            try {
                ClassificationResult result;
                result.filename = image_files[i].filename().string();
                result.score = calculateScore(image_files[i].string());
                result.predicted = (result.score > threshold);
                result.actual = labels[i] > 0;
                result.correct = (result.predicted == result.actual);
                results.push_back(result);
                if ((i + 1) % 50 == 0) {
                    std::cout << "Elaborate " << (i + 1) << " immagini..." << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Errore in " << image_files[i].filename() 
                         << ": " << e.what() << std::endl;
            }
        }
        return results;
    }
    
    Metrics calculateMetrics(const std::vector<ClassificationResult>& results) {
        Metrics m;
        for (const auto& r : results) {
            if (r.predicted && r.actual) m.true_positives++;
            else if (r.predicted && !r.actual) m.false_positives++;
            else if (!r.predicted && r.actual) m.false_negatives++;
            else m.true_negatives++;
        }
        return m;
    }
    
    void setThreshold(float new_threshold) { threshold = new_threshold; }
    float getThreshold() const { return threshold; }
};

// stampa risultati (identico al tuo)
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
    
    for (size_t i = 0; i < std::min<size_t>(10, results.size()); ++i) {
        const auto& r = results[i];
        std::cout << std::setw(15) << r.filename.substr(0, 15) << "  "
                  << std::setw(7) << r.score << "  "
                  << std::setw(4) << (r.predicted ? "POS" : "NEG") << "  "
                  << std::setw(6) << (r.actual ? "POS" : "NEG") << "  "
                  << (r.correct ? "✓" : "✗") << std::endl;
    }
}

// --- (opzionale) auto-tuning soglia come funzione riutilizzabile ---
static float findBestThreshold(const std::vector<ClassificationResult>& results) {
    std::vector<float> scores; scores.reserve(results.size());
    for (const auto& r : results) scores.push_back(r.score);
    std::sort(scores.begin(), scores.end());
    scores.erase(std::unique(scores.begin(), scores.end()), scores.end());
    if (scores.empty()) return 0.5f;

    auto eval_at = [&](float th){
        Metrics m{};
        for (const auto& r : results) {
            bool pred = (r.score > th);
            if (pred && r.actual) m.true_positives++;
            else if (pred && !r.actual) m.false_positives++;
            else if (!pred && r.actual) m.false_negatives++;
            else m.true_negatives++;
        }
        float J = m.sensitivity() + m.specificity() - 1.0f;
        return std::make_pair(J, m);
    };

    float best_th = scores.front(), best_J = -1e9f;
    for (float th : scores) {
        auto [J, _] = eval_at(th);
        if (J > best_J) { best_J = J; best_th = th; }
    }
    return best_th;
}

static void write_results_csv(const std::string& path, const std::vector<ClassificationResult>& results){
    std::ofstream ofs(path);
    ofs <<"filename,score;pred,actual,correct\n";
    for (const auto& r: results){
        ofs <<r.filename << "," <<r.score << ","
            << (r.predicted ? 1 : 0) << ","
            << (r.actual ? 1 : 0) << ","
            << (r.correct ? 1 : 0) <<"\n";
    }
}

static void write_metrics_txt(const std:: string& path, const Metrics& m, float thr){
    std::ofstream ofs(path);
    ofs << "Threshold: " << thr << "\n";
    ofs << "Accuracy: "    << m.accuracy()*100     << "%\n";
    ofs << "Sensitivity: " << m.sensitivity()*100  << "%\n";
    ofs << "Specificity: " << m.specificity()*100  << "%\n";
    ofs << "Precision: "   << m.precision()*100    << "%\n\n";
    ofs << "Confusion Matrix (TN FP / FN TP):\n";
    ofs << m.true_negatives << " " << m.false_positives << "\n"
        << m.false_negatives << " " << m.true_positives << "\n";


}

int main(int argc, char** argv) {
    try {
        // Usa i pesi statici generati
        std::string weights_file = "../data/weights/static_weights_224x224.npy";
        std::string labels_file  = "../data/ChestMNIST_Images/labels.npy";
        std::string images_dir   = "../data/ChestMNIST_Images";
        
        //output di default
        std::string out_csv="../results/sequential_scores.csv";
        std::string out_metrics="../results/sequential_metrics.txt";

        //parametri
        bool auto_threshold=true;
        float threshold= 0.0f;
        size_t limit= 0;

        //parse argomenti
        for (int i=1; i<argc; ++i){
            if(!std::strcmp(argv[i], "--weights") && i+1 < argc)
                weights_file = argv[++i];
            else if (!std::strcmp(argv[i], "--labels") && i+1 <argc) labels_file = argv[++i];
            else if (!std::strcmp(argv[i], "--images") && i+1 < argc)   images_dir = argv[++i];
            else if (!std::strcmp(argv[i], "--limit")  && i+1 < argc)   limit = std::stoul(argv[++i]);
            else if (!std::strcmp(argv[i], "--out-csv") && i+1 < argc)  out_csv = argv[++i];
            else if (!std::strcmp(argv[i], "--out-metrics") && i+1 < argc) out_metrics = argv[++i];
            else if (!std::strcmp(argv[i], "--auto-th"))                auto_threshold = true;
            else if (!std::strcmp(argv[i], "--threshold") && i+1 < argc){ auto_threshold = false; threshold = std::stof(argv[++i]); }
        }
        // se serve, crea cartella results
        try
        {
            std::filesystem::create_directories(std::filesystem::path(out_csv).parent_path());
        }
        catch(...){}
        
        try { std::filesystem::create_directories(std::filesystem::path(out_metrics).parent_path()); } catch(...) {}

        

        //istanzia classificatore
        CardiomegalyClassifier classifier(weights_file, labels_file, threshold);
       
        //classifica (con eventuale limit)
        auto all_results = classifier.classifyAll(images_dir);

        std::vector<ClassificationResult> results;

        results.reserve(all_results.size());

        if (limit > 0){
            for (size_t i = 0; i < std::min(limit, all_results.size()); ++i) results.push_back(all_results[i]);
        } else {
            results = std::move(all_results);
        }

        //soglia
        float used_threshold = threshold;
        if (auto_threshold){
            used_threshold = findBestThreshold(results);
            classifier.setThreshold(used_threshold);
        }

        //applicazione soglia scelta
        for (auto& r : results){
            r.predicted = (r.score > used_threshold);
            r.correct = (r.predicted == r.actual);
        }


        auto metrics = classifier.calculateMetrics(results);

        //Stampa a console
        printResults(results, metrics);

        //salva su file
        write_results_csv(out_csv, results);
        write_metrics_txt(out_metrics, metrics, used_threshold);

        std::cout << "\nSalvati:\n  " << out_csv << "\n  " << out_metrics << std::endl;
        return 0;
    }catch (const std::exception& e){
        std::cerr << "Errore: " <<e.what() <<std::endl;
        return 1;
    }

}
