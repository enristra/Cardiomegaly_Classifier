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

using namespace std;
namespace fs = std::filesystem;


struct ClassificationResult {
    string filename;
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
    vector<float> weights;   // 224*224
    vector<int> labels;      // N
    float threshold;
    
public:
    CardiomegalyClassifier(const string& weights_file, const string& labels_file, float classification_threshold = 0.5f) 
        : threshold(classification_threshold) {
        loadWeights(weights_file);
        loadLabels(labels_file);
    }
    
    void loadWeights(const string& filename) {
        cout << "Caricamento pesi da: " << filename << endl;
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        
        if (arr.shape.size() != 2 || arr.shape[0] != 224 || arr.shape[1] != 224) {
            throw runtime_error("I pesi devono essere 224x224");
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
            throw runtime_error("Tipo di peso non supportato");
        }

        // Normalizza i pesi in modo che sommino a 1 (media pesata dei pixel)
        double s = 0.0;
        for (float w : weights) s += w;
        if (s > 0) {
            for (float& w : weights) w = static_cast<float>(w / s);
            cout << "Pesi normalizzati: somma = 1.0\n";
        } else {
            cout << "Attenzione: somma pesi = 0, nessuna normalizzazione applicata.\n";
        }
        float wmin = *min_element(weights.begin(), weights.end());
float wmax = *max_element(weights.begin(), weights.end());
cout << "Peso min: " << wmin << ", max: " << wmax << endl;
        
        cout << "Pesi caricati: " << weights.size() << " elementi" << endl;
    }
    
    void loadLabels(const string& filename) {
        cout << "Caricamento labels da: " << filename << endl;
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        
        if (arr.shape.size() != 2 || arr.shape[1] != 14) {
            throw runtime_error("Le labels devono essere (N,14) come ChestMNIST.");
        }

        size_t num_samples = arr.shape[0];
        size_t label_index_cardiomegaly = 1; // cardiomegalia

        labels.clear();
        labels.reserve(num_samples);

        if (arr.word_size == sizeof(uint8_t)) {
        uint8_t* data = arr.data<uint8_t>();
        for (size_t i = 0; i < num_samples; ++i) {
            labels.push_back(static_cast<int>(data[i * arr.shape[1] + label_index_cardiomegaly]));
        }
        } else {
            throw runtime_error("Tipo di label non supportato in labels.npy.");
        }
        
        cout << "Labels caricate: " << labels.size() << " elementi (solo cardiomegalia)" << endl;
        // --- DEBUG: conteggio positivi/negativi ---
    int num_positivi = 0;
    int num_negativi = 0;
    for (int l : labels) {
        if (l > 0) num_positivi++;
        else num_negativi++;
    }
    cout << "Numero di campioni positivi: " << num_positivi << endl;
    cout << "Numero di campioni negativi: " << num_negativi << endl;
}
    
    float calculateScore(const string& image_file) {
        cnpy::NpyArray arr = cnpy::npy_load(image_file);
        
        vector<float> image_data;
        
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
            throw runtime_error("Tipo di immagine non supportato");
        }
        
        size_t pixels_per_channel = 224 * 224;
        if (image_data.size() % pixels_per_channel != 0) {
            throw runtime_error("Dimensioni immagine non valide");
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
    
    vector<ClassificationResult> classifyAll(const string& images_dir) {
        vector<ClassificationResult> results;
        
        vector<fs::path> image_files;
        for (const auto& entry : fs::directory_iterator(images_dir)) {
            if (entry.path().extension() == ".npy" && 
                entry.path().filename().string().find("image_") == 0) {
                image_files.push_back(entry.path());
            }
        }
        sort(image_files.begin(), image_files.end());
        
        cout << "Classificazione di " << image_files.size() << " immagini..." << endl;
        size_t max_n = min(image_files.size(), labels.size());
        if (max_n == 0){
            cerr << "Nessuna coppia (immagine, label) trovata\n";
            return{};
        }

        for (size_t i = 0; i < max_n; ++i) {
            try {
                ClassificationResult result;
                result.filename = image_files[i].filename().string();
                result.score = calculateScore(image_files[i].string());
                result.predicted = (result.score > threshold);
                result.actual = labels[i] > 0;
                result.correct = (result.predicted == result.actual);
                results.push_back(result);
                if ((i + 1) % 50 == 0) {
                    cout << "Elaborate " << (i + 1) << " immagini..." << endl;
                }
            } catch (const exception& e) {
                cerr << "Errore in " << image_files[i].filename() 
                         << ": " << e.what() << endl;
            }
        }
        return results;
    }
    
    Metrics calculateMetrics(const vector<ClassificationResult>& results) {
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
void printResults(const vector<ClassificationResult>& results, const Metrics& metrics) {
    cout << "\n=== RISULTATI CLASSIFICAZIONE ===" << endl;
    cout << fixed << setprecision(3);
    
    cout << "\nMETRICHE:" << endl;
    cout << "Accuracy:    " << metrics.accuracy() * 100 << "%" << endl;
    cout << "Sensitivity: " << metrics.sensitivity() * 100 << "%" << endl;
    cout << "Specificity: " << metrics.specificity() * 100 << "%" << endl;
    cout << "Precision:   " << metrics.precision() * 100 << "%" << endl;
    
    cout << "\nCONFUSION MATRIX:" << endl;
    cout << "                 Predicted" << endl;
    cout << "                 Neg   Pos" << endl;
    cout << "Actual    Neg   " << setw(4) << metrics.true_negatives 
              << "  " << setw(4) << metrics.false_positives << endl;
    cout << "          Pos   " << setw(4) << metrics.false_negatives 
              << "  " << setw(4) << metrics.true_positives << endl;
    
    cout << "\nPRIMI 10 RISULTATI:" << endl;
    cout << "File                Score    Pred  Actual  Correct" << endl;
    cout << "------------------------------------------------" << endl;
    
    for (size_t i = 0; i < min<size_t>(10, results.size()); ++i) {
        const auto& r = results[i];
        cout << setw(15) << r.filename.substr(0, 15) << "  "
                  << setw(7) << r.score << "  "
                  << setw(4) << (r.predicted ? "POS" : "NEG") << "  "
                  << setw(6) << (r.actual ? "POS" : "NEG") << "  "
                  << (r.correct ? "✓" : "✗") << endl;
    }
}

//auto-tuning soglia come funzione riutilizzabile ---
static float findBestThreshold(const vector<ClassificationResult>& results) {
    vector<float> scores; scores.reserve(results.size());
    for (const auto& r : results) scores.push_back(r.score);
    sort(scores.begin(), scores.end());
    scores.erase(unique(scores.begin(), scores.end()), scores.end());
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
        return make_pair(J, m);
    };

    float best_th = scores.front(), best_J = -1e9f;
    for (float th : scores) {
        auto [J, _] = eval_at(th);
        if (J > best_J) { best_J = J; best_th = th; }
    }
    return best_th;
}

static void write_results_csv(const string& path, const vector<ClassificationResult>& results){
    ofstream ofs(path);
    ofs <<"filename,score,pred,actual,correct\n";
    for (const auto& r: results){
        ofs <<r.filename << "," <<r.score << ","
            << (r.predicted ? 1 : 0) << ","
            << (r.actual ? 1 : 0) << ","
            << (r.correct ? 1 : 0) <<"\n";
    }
}

static void write_metrics_txt(const  string& path, const Metrics& m, float thr){
    ofstream ofs(path);
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
        string weights_file = "../data/weights/cardio_weights_224x224.npy";
        string labels_file  = "../data/labels.npy";
        string images_dir   = "../data/images";
        
        //output di default
        string out_csv="../results/sequential_scores.csv";
        string out_metrics="../results/sequential_metrics.txt";

        //parametri
        bool auto_threshold=true;
        float threshold= 0.0f;
        size_t limit= 0;

        //parse argomenti
        for (int i=1; i<argc; ++i){
            if(!strcmp(argv[i], "--weights") && i+1 < argc)
                weights_file = argv[++i];
            else if (!strcmp(argv[i], "--labels") && i+1 <argc) labels_file = argv[++i];
            else if (!strcmp(argv[i], "--images") && i+1 < argc)   images_dir = argv[++i];
            else if (!strcmp(argv[i], "--limit")  && i+1 < argc)   limit = stoul(argv[++i]);
            else if (!strcmp(argv[i], "--out-csv") && i+1 < argc)  out_csv = argv[++i];
            else if (!strcmp(argv[i], "--out-metrics") && i+1 < argc) out_metrics = argv[++i];
            else if (!strcmp(argv[i], "--auto-th"))                auto_threshold = true;
            else if (!strcmp(argv[i], "--threshold") && i+1 < argc){ auto_threshold = false; threshold = stof(argv[++i]); }
        }
        // se serve, crea cartella results
        try
        {
            filesystem::create_directories(filesystem::path(out_csv).parent_path());
        }
        catch(...){}
        
        try { filesystem::create_directories(filesystem::path(out_metrics).parent_path()); } catch(...) {}

        

        //istanzia classificatore
        CardiomegalyClassifier classifier(weights_file, labels_file, threshold);
       
        //classifica (con eventuale limit)
        auto all_results = classifier.classifyAll(images_dir);

        vector<ClassificationResult> results;

        results.reserve(all_results.size());

        if (limit > 0){
            for (size_t i = 0; i < min(limit, all_results.size()); ++i) results.push_back(all_results[i]);
        } else {
            results = move(all_results);
        }

        // --- DEBUG: statistiche score
        float min_score = numeric_limits<float>::max();
        float max_score = numeric_limits<float>::lowest();
        vector<float> scores;
        scores.reserve(results.size());

        //applicazione soglia scelta
        for (auto& r : results){
            scores.push_back(r.score);
            min_score = min(min_score, r.score);
            max_score = max(max_score, r.score);
        }

        sort(scores.begin(), scores.end());
        float median_score = scores[scores.size()/2];

        cout << "Score min: " << min_score << ", max: " << max_score 
     << ", median: " << median_score << endl;

        // --- Soglia basata sulla mediana o fissa ---
        //float used_threshold = 0.5f;        // soglia fissa
        float used_threshold = median_score; // soglia = mediana


        classifier.setThreshold(used_threshold);

        //Applicazione soglia scelta
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

        cout << "\nSalvati:\n  " << out_csv << "\n  " << out_metrics << endl;
        return 0;
    }catch (const exception& e){
        cerr << "Errore: " <<e.what() <<endl;
        return 1;
    }

}
