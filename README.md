# Skin Lesion Classification Using CNN

A complete deep learning pipeline for **multi-class classification of skin cancer lesions** using Convolutional Neural Networks (CNNs). Includes data preprocessing, model building, training, evaluation, and interpretation — designed for transparency and reproducibility.

## Dataset

The [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) dataset contains 10,015 dermatoscopic images classified into 7 categories:

| Class | Description |
|-------|-------------|
| nv | Melanocytic nevi |
| mel | Melanoma |
| bkl | Benign keratosis-like lesions |
| bcc | Basal cell carcinoma |
| akiec | Actinic keratoses |
| vasc | Vascular lesions |
| df | Dermatofibroma |

Images were resized to **100 x 75 x 3** to maintain visual quality while optimizing training speed.

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 77.7% |
| Test Accuracy | 65.9% |
| Best Class Precision | 0.79 (Melanocytic nevi) |
| Lowest Class Recall | 0.00 (rare classes) |

## Pipeline Overview

1. **Data Loading & Preprocessing** — Metadata merged with image paths, images resized, labels one-hot encoded, data split into train/val/test
2. **Exploratory Data Analysis** — Class distribution plotted, imbalance identified
3. **CNN Model Construction** — Sequential model with two convolutional blocks, dropout regularization, data augmentation via `ImageDataGenerator`
4. **Training** — 50 epochs with Adam optimizer and `ReduceLROnPlateau` callback
5. **Evaluation** — Confusion matrices, classification reports, learning curves

## Getting Started

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Wobbly1212/CNN_Skin-Cancer-Classifier.git
cd CNN_Skin-Cancer-Classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
2. Place the images and metadata CSV in a `data/HAM10000/` directory
3. Update the data path in the notebook if needed

### Running the Notebook

```bash
jupyter notebook skin_cancer_project.ipynb
```

## Project Structure

```
CNN_Skin-Cancer-Classifier/
├── skin_cancer_project.ipynb   # Main analysis notebook
├── skin_cancer_project.pdf     # Full written report
├── requirements.txt            # Python dependencies
├── LICENSE
└── README.md
```

## Dependencies

- TensorFlow / Keras
- scikit-learn
- NumPy, Pandas
- Seaborn, Matplotlib
- Pillow

## Limitations & Future Work

- Low recall on rare classes (melanoma, dermatofibroma) due to class imbalance
- Future improvements:
  - Transfer learning (EfficientNet, ResNet)
  - Focal loss or class weighting
  - Oversampling / SMOTE techniques

## Author

Developed by **Diako Darabi** as part of the Data Science Master's Program.

## License

This project is licensed under the [MIT License](LICENSE).
