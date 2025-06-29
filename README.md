# Quora Duplicate Question Detection

## 📄 Project Overview

This project builds a deep learning model to detect **duplicate questions** in the Quora Question Pairs dataset (404K+ question pairs). By identifying semantically similar questions, this model helps reduce redundant content and improve user experience on Q\&A platforms.

---

## 💡 Problem Statement

Online forums like Quora often have multiple users asking the same question phrased differently. Automatically detecting these duplicates helps merge answers, avoid clutter, and surface the most relevant responses.

---

## 🗂️ Dataset

* **Source:** Quora Question Pairs dataset
* **Size:** 404,000+ question pairs
* **Labels:** 1 (duplicate) or 0 (not duplicate)

---

## ⚙️ Approach

### ✅ Preprocessing

* Applied **Tokenization**, **Positional Embeddings**, and **Attention Masks** using Hugging Face Transformers.
* Handled variable-length sequences with dynamic padding and attention masks.

### ✅ Model Architecture

* Implemented a **Siamese BERT network** using two parallel BERT encoders with shared weights.
* Generated sentence embeddings using the BERT \[CLS] token representations.
* Compared embeddings using a **Manhattan distance layer** followed by dense layers for final classification.

### ✅ Training Details

* Optimizer: Adam with learning rate 1e-5
* Loss: Binary Crossentropy
* Additional regularization using dropout and pooling layers

---

## 🚀 Results

* **Validation Accuracy:** 86.7%
* **F1-score:** 0.88
* Reduced false duplicate predictions by **17%**, leading to more accurate content merging.

---

## 💻 Tech Stack

* **Python**, **PyTorch**, **TensorFlow**, **Keras**, **Hugging Face Transformers**
* Google Colab for accelerated training using GPU

---

## ✨ Future Work

* Experiment with other transformer models like RoBERTa and DeBERTa.
* Deploy as an API to integrate into live Q\&A platforms.
* Explore contrastive loss instead of binary cross-entropy for more robust representation learning.

---

## 📝 How to Run

```bash
# Install required packages
pip install transformers tensorflow

# Run the training script
python train.py
```

---

## 🤝 Acknowledgements

* Hugging Face Transformers for pre-trained BERT
* Quora for providing a rich real-world dataset
* Original community contributions and academic papers on Siamese networks

---

## 📬 Contact

For questions or collaborations, feel free to reach out via \[your email or LinkedIn].
