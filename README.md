# 🫀 Cardiomegaly Classifier | Exploring CUDA for Medical Imaging

A simple, **educational project** developed during my Master's in Computer Engineering at the University of Bologna.  
The goal: explore **GPU programming with CUDA** by optimizing inference for a basic, interpretable classifier applied to chest X-ray images (CHESTMNIST dataset).

---

## 📌 About this project
This work is **purely didactic**: it does not aim to provide a real clinical solution, but to experiment with:
- Applying **AI concepts** to medical imaging.
- Implementing a **sequential version (C++)** and a **parallel version (CUDA)**.
- Understanding **GPU memory management, thread configuration, and performance bottlenecks**.

---

## 🎯 Objectives
- Classify grayscale images (224×224) from CHESTMNIST.
- Compare **CPU vs GPU** performance during inference.
- Learn fundamentals of:
  ✅ GPU programming  
  ✅ Shared vs global memory  
  ✅ Reduction and parallelization patterns  

---

## 🧠 Methodology
- **Classifier logic**: weighted sum of pixels, weights obtained from a simple linear model trained in PyTorch.
- **Inference process**:
  - Dot product (pixel × weight)
  - Threshold-based binary classification (cardiomegaly: yes/no)

---

## 📊 Expected Results
- **Speedup** analysis: CUDA vs sequential C++.
- **Performance tuning**: memory optimizations, grid/block configuration.
- **Diagnostic metrics**: accuracy, sensitivity, specificity, precision.

---

## 📚 Dataset
- [CHESTMNIST – MedMNIST v2](https://medmnist.com/)
- Grayscale images, size **224×224 px**
- Target class: Cardiomegaly (index 5 in multilabel setting)

---

## 👨‍💻 Author
**Enrico Strangio**  
MSc in Computer Engineering @ University of Bologna  
🔗 [LinkedIn](https://www.linkedin.com/in/enricostrangio) | [GitHub](https://github.com/enristra)

---

## 🧭 Status
🚧 Work in Progress – CUDA optimization and documentation ongoing  
🔍 Shared for **educational purposes only**

---

## 🔗 Related Links
- Project showcase on LinkedIn: *coming soon*
