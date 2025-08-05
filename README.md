# 🫀 Cardiomegaly Classifier | Exploring CUDA for Medical Imaging

A simple classifier for detecting cardiomegaly from chest X-rays, developed as part of the Digital Systems course at the University of Bologna.

The main goal is to optimize inference using CUDA, applying parallel programming concepts to a basic, interpretable model instead of complex deep neural networks.
---

## 🎯 Objectives
Classify grayscale images (224×224) from CHESTMNIST dataset.

Compare performance between a sequential C++ implementation and a parallel CUDA version.

Learn and apply key concepts in:
✅ GPU programming
✅ Memory and thread management
✅ Parallelization of numerical algorithms
---

## 🧠 Methodology
The algorithm is based on a weighted sum of pixels, with weights learned from a simple linear model trained in PyTorch.

Inference consists of a dot product (pixel × weight) followed by a threshold-based decision for binary classification (presence or absence of cardiomegaly).
---

## 📊 Expected Results
Speedup measurement: GPU vs CPU during inference.

Optimal threshold selection based on accuracy, sensitivity, and precision.

Performance analysis: bottlenecks and opportunities for further optimization (shared memory, grid/block configuration).

---

## 📚 Dataset

- Dataset: [CHESTMNIST – MedMNIST v2](https://medmnist.com/)
- Grayscale images, 224×224 pixel.
- Target class: **Cardiomegaly** (indice 5 su 14 etichette multilabel).

---

## 👨‍💻 Author

**Enrico Strangio**  
Computer Engineering MSc @Unibo  
[LinkedIn](https://www.linkedin.com/in/enrico-strangio/) – [GitHub](https://github.com/enristra)

---

## 🧭 Project Status

🚧 Work in Progress
CUDA optimizations and documentation are ongoing.
Shared for educational purposes only.
