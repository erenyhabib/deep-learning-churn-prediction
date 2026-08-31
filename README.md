# 📊 Customer Churn Prediction using Regularized Deep Learning

An end-to-end Binary Classification Deep Learning pipeline built with **TensorFlow/Keras** to predict customer churn. The project focuses on handling class imbalance, preventing data leakage, and resolving model overfitting through **Dropout Regularization** and **EarlyStopping**.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Model Architecture](#-model-architecture)
- [Overfitting Mitigation & Training](#-overfitting-mitigation--training)
- [Model Evaluation & Results](#-model-evaluation--results)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Technologies Used](#-technologies-used)

---

## 💡 Project Overview
Customer churn is a critical metric for businesses. This project builds a Deep Learning neural network designed to identify customers at risk of leaving. The primary technical objective was achieving strong **generalization performance** on unseen test data without overfitting to the training set.

---

## ⚙️ Dataset & Preprocessing
- **Dataset Size:** 7,000+ customer records.
- **Split Strategy:** Strict 3-way split to completely eliminate data leakage:
  - **70%** Training Set
  - **15%** Validation Set
  - **15%** Isolated Test Set
- **Scaling & Encoding:** Features were scaled using `StandardScaler` and categorical variables encoded via `One-Hot Encoding`.

---

## 🏗️ Model Architecture
The network is built using the Keras `Sequential` API:

1. **Input Layer:** Takes normalized feature vectors.
2. **Hidden Layer 1:** `Dense` (32 units, `ReLU` activation) + `Dropout(0.3)`.
3. **Hidden Layer 2:** `Dense` (16 units, `ReLU` activation) + `Dropout(0.2)`.
4. **Output Layer:** `Dense` (1 unit, `Sigmoid` activation for binary classification).

- **Optimizer:** `Adam`
- **Loss Function:** `Binary Cross-Entropy`
- **Metrics:** `Accuracy`

---

## 🛡️ Overfitting Mitigation & Training
Initial unregularized training resulted in noticeable overfitting (Train Acc ~83% vs Val Acc ~79%). To resolve this:

* **Dropout Layers:** Prevents neuron co-adaptation by randomly dropping out 30% and 20% of neurons during training cycles.
* **EarlyStopping Callback:** Dynamically monitored `val_loss` with `patience=10` and automatically restored the optimal model weights (`restore_best_weights=True`).

---

## 📈 Model Evaluation & Results

### 1. Generalization Across Datasets
The regularized model converged smoothly around epoch 17, demonstrating near-perfect alignment across all three subsets:

| Metric | Training Set | Validation Set | Isolated Test Set |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **80.73%** | **79.18%** | **78.77%** |
| **Loss** | **0.4108** | **0.4244** | **0.4369** |

> **Key Takeaway:** The margin between Training Accuracy and Test Accuracy is **< 2%**, proving that overfitting was successfully eliminated.

### 2. Classification Report (Test Set)
- **Overall Test Accuracy:** 78.77%
- **Class 0 (No Churn):** Precision `0.84` | Recall `0.87` | F1-Score `0.86`
- **Class 1 (Churn):** Precision `0.61` | Recall `0.55` | F1-Score `0.58`

---

## 📂 Repository Structure

```text
├── churn_prediction.ipynb     # Main Jupyter Notebook with step-by-step code
├── requirements.txt           # Required Python packages
└── README.md                  # Project documentation
