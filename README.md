# 🫀 Cardiomegaly Classifier – CHESTMNIST + CUDA

Un semplice classificatore per la rilevazione della **cardiomegalia** da radiografie toraciche, sviluppato per l’esame di *Sistemi Digitali* presso l’Università di Bologna.

Il progetto si concentra sull’**ottimizzazione della fase di inferenza** tramite **CUDA**, utilizzando un modello interpretabile e lineare, senza reti neurali complesse.

---

## 🎯 Obiettivi

- Classificare immagini 224×224 in scala di grigi (CHESTMNIST).
- Confrontare le prestazioni tra una versione sequenziale in C++ e una versione parallela in CUDA.
- Apprendere concetti fondamentali di:
  - programmazione GPU,
  - gestione della memoria e dei thread,
  - parallelizzazione di algoritmi numerici.

---

## 🧠 Metodo

L’algoritmo si basa su una **somma pesata dei pixel**, con i pesi appresi da un **modello lineare addestrato in PyTorch**.  
L'inferenza consiste in un semplice prodotto scalare `pixel × peso`, seguito da un confronto con soglia per la classificazione binaria (presenza o assenza di cardiomegalia).

---

## 📊 Risultati previsti

- Misura dello **speedup GPU vs CPU** nella fase di inferenza.
- Valutazione della **soglia decisionale ottimale** basata su accuratezza, sensibilità e precisione.
- Analisi dei colli di bottiglia computazionali e delle potenzialità di parallelizzazione.

---

## 📚 Dataset

- Dataset utilizzato: [CHESTMNIST – MedMNIST v2](https://medmnist.com/)
- Immagini in scala di grigi, 224×224 pixel.
- Classe target: **Cardiomegaly** (indice 5 su 14 etichette multilabel).

---

## 👨‍💻 Autore

**Enrico Strangio**  
Computer Engineering MSc @Unibo  
[LinkedIn](https://www.linkedin.com/in/enrico-strangio/) – [GitHub](https://github.com/enristra)

---

## 🧭 Stato del progetto

🚧 In fase di sviluppo – documentazione e ottimizzazioni CUDA in corso.  
Condiviso pubblicamente a scopo didattico e di apprendimento.
