
# 🧬 Skin Lesion Classification Using CNN

This project presents a complete deep learning pipeline for **multi-class classification of skin cancer lesions** using Convolutional Neural Networks (CNNs). It includes data preprocessing, model building, training, evaluation, and interpretation — all designed for high transparency and reproducibility.

---

## 📁 Dataset

We used the **HAM10000** dataset, which contains 10,015 dermatoscopic images classified into 7 categories:

1. Melanocytic nevi
2. Melanoma
3. Benign keratosis-like lesions
4. Basal cell carcinoma
5. Actinic keratoses
6. Vascular lesions
7. Dermatofibroma

Images were resized to **100×75×3** to maintain visual quality while optimizing for training speed.

---

## 🧪 Project Pipeline Overview

### ✅ Step 1: Data Loading & Preprocessing
- Metadata loaded and merged with image paths
- Images resized
- Target labels encoded as integers and one-hot vectors
- Dataset split into train, validation, and test sets

### 🏷️ Step 2: Label Encoding
- One-hot encoding using `to_categorical` for multi-class classification

### 📊 Step 3: Exploratory Data Analysis
- Class distribution plotted
- Class imbalance observed, especially for rare lesions

### 🧠 Step 4–6: CNN Model Construction
- Sequential CNN model with two convolutional blocks and fully connected layers
- Dropout for regularization
- Data augmentation using `ImageDataGenerator`
- Optimized using Adam with `ReduceLROnPlateau` callback

### 📈 Step 7: Training the Model
- Trained for 50 epochs
- Tracked training/validation accuracy and loss
- Plotted learning curves to assess performance

### 🧪 Step 8: Model Evaluation
- Evaluated on test and validation sets
- Confusion matrices and classification reports generated
- Performance metrics interpreted by class

### 📉 Step 9: Quantitative Performance Summary
- Final performance:
  - **Validation Accuracy:** 77.7%
  - **Test Accuracy:** 65.9%
  - Good performance on dominant classes (e.g., Melanocytic nevi)
  - Low recall on minority classes due to class imbalance

---

## 🖼️ Visualization Examples

| Class                  | Sample |
|------------------------|--------|
| Melanoma               | *(sample image)* |
| Benign keratosis       | *(sample image)* |
| Melanocytic nevi       | *(sample image)* |

*(Add representative samples when possible.)*

---

## ⚙️ Dependencies

- Python 3.12
- TensorFlow / Keras
- scikit-learn
- NumPy, Pandas, Seaborn, Matplotlib
- PIL (Python Imaging Library)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 💾 Model Saving

Model saved using:
```python
model.save("model.h5")  # HDF5 format (legacy)
```
Recommended format:
```python
model.save("model.keras")  # Native Keras format
```

---

## 🔍 Limitations & Future Work

- Low recall on rare classes like melanoma and dermatofibroma
- Imbalanced dataset heavily skews predictions toward dominant class
- Future improvements:
  - Apply transfer learning (e.g., EfficientNet, ResNet)
  - Use focal loss or class weighting
  - Apply oversampling or SMOTE techniques

---

## 📄 Final Performance Metrics

| Metric                | Value      |
|-----------------------|------------|
| Validation Accuracy   | 77.7%      |
| Test Accuracy         | 65.9%      |
| Best Class Precision  | 0.79 (Melanocytic nevi) |
| Lowest Class Recall   | 0.00 (for rare classes) |

---

## 📘 PDF Report

A full report with methodology, code breakdown, and performance analysis is included:  
📎 [`skin_cancer_project.pdf`](skin_cancer_project.pdf)

---

## 👨‍💻 Author

Developed by **[Mohammadhossein Darabi]**  
As part of the Data Science Master's Program  
⭐️ Star this repo if you found it helpful!
