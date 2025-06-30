// src/utils.hpp
#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <iostream>
#include <stdexcept> // Per std::runtime_error e std::exception

// Includi cnpy.h - Assicurati che cnpy.h e cnpy.cpp siano nella cartella src/
#include "cnpy.h"

// Funzione per caricare un array 2D da un file .npy
template<typename T>
std::vector<T> load_npy_data(const std::string& filepath, int expected_rows, int expected_cols) {
    try {
        cnpy::NpyArray arr = cnpy::npy_load(filepath);

        // Verifica le dimensioni
        if (arr.shape.size() != 2 || arr.shape[0] != expected_rows || arr.shape[1] != expected_cols) {
            std::string error_msg = "Dimensioni del file NPY non corrispondenti per: " + filepath +
                                    ". Attese: [" + std::to_string(expected_rows) + ", " + std::to_string(expected_cols) + "]" +
                                    ", Trovate: [" + (arr.shape.size() > 0 ? std::to_string(arr.shape[0]) : "N/A") +
                                    ", " + (arr.shape.size() > 1 ? std::to_string(arr.shape[1]) : "N/A") + "]";
            throw std::runtime_error(error_msg);
        }
        
        // Verifica il tipo di dati usando 'word_size'
        // 'word_size' è la dimensione in byte di un singolo elemento nel file NPY.
        // Questo è corretto e presente nel tuo cnpy.h
        if (arr.word_size != sizeof(T)) {
             std::cerr << "Attenzione: Dimensione del tipo di dati del file NPY (" << arr.word_size << " bytes) non corrisponde al tipo atteso (" << sizeof(T) << " bytes) per " << filepath << std::endl;
        }

        // Accedi ai dati grezzi.
        // La tua cnpy.h ha un metodo template 'data()' (con parentesi) che restituisce T*.
        // E ha il membro 'num_vals' che è il numero totale di elementi.
        std::vector<T> data(arr.data<T>(), arr.data<T>() + arr.num_vals); // <-- Questo è corretto per il tuo cnpy.h

        return data;

    // Cattura l'eccezione generica std::runtime_error, dato che cnpy::npy_error non è definita.
    // La funzione npy_load in questa versione di cnpy lancia std::runtime_error.
    } catch (const std::runtime_error& e) { // <-- Questo è il catch corretto per la tua cnpy.h
        std::cerr << "Errore nel caricamento del file NPY " << filepath << ": " << e.what() << std::endl;
        throw; // Rilancia l'eccezione per gestirla a un livello superiore se necessario
    } catch (const std::exception& e) { // Cattura altre eccezioni standard generiche
        std::cerr << "Errore generico nel caricamento del file NPY " << filepath << ": " << e.what() << std::endl;
        throw;
    }
}

#endif // UTILS_HPP