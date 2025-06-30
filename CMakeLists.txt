# CMakeLists.txt nella directory principale del progetto Cardiomegaly_Classifier

# Specifica la versione minima di CMake richiesta
cmake_minimum_required(VERSION 3.10)

# Definisci il nome del progetto e i linguaggi usati
# LANGUAGES CXX indica C++
project(CardiomegalyClassifier LANGUAGES CXX)

# Imposta lo standard C++ a C++17
# CXX_STANDARD_REQUIRED ON forza il compilatore a usare C++17 e fallisce se non può
# CXX_EXTENSIONS OFF disabilita le estensioni specifiche del compilatore per maggiore portabilità
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Aggiungi le directory di inclusione dove il compilatore cercherà i file .hpp
# ${CMAKE_CURRENT_SOURCE_DIR}/src punta alla tua cartella 'src/'
# Questo è dove si trovano common.hpp, utils.hpp e cnpy.h
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Aggiungi il target per l'eseguibile sequenziale (versione CPU)
# Indica a CMake di creare un eseguibile chiamato sequential_classifier
# usando i file sorgente specificati.
# src/cnpy.cpp è incluso qui perché hai copiato i file cnpy in src/.
add_executable(sequential_classifier
    src/main_sequential.cpp
    src/cnpy.cpp
)

# Trova e collega la libreria zlib (necessaria per cnpy)
find_package(ZLIB REQUIRED)
target_link_libraries(sequential_classifier PRIVATE ZLIB::ZLIB)
