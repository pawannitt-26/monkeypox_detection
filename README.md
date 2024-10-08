# Monkeypox Detection Model

This project aims to develop a machine learning model for detecting **Monkeypox**, **Chickenpox**, and other skin diseases like **Measles** and **Normal** (healthy skin) using image classification techniques. The model leverages data preprocessing, augmentation, and a combination of machine learning and deep learning approaches for classification.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

---

## Project Structure

The project is structured to ensure modularity, scalability, and maintainability. Below is the overview of the folder structure:

```
monkeypox_detection/
├── notebooks/                        # Jupyter notebooks for experiments
│   ├── deep_learning_model           # Deep learning approach
│   ├── machine_learning_model        # Machine learning approaches (KNN, SVM, XGBoost)
│   └── proposed_model                # Combination and hybrid model
|
├── results/                          # Results directory
│   ├── models/                       # Saved model checkpoints
│   └── plots/                        # Plots of training curves, confusion matrix, etc.
│
├── src/                          # Source code
│   ├── data/                     # Dataset directory
│   │   ├── raw/                  # Raw dataset before processing
│   │   ├── train/                # Training data split
│   │   ├── validation/           # Validation data split
│   │   └── test/                 # Test data split
│   └── config.py                 # Configuration settings like paths and hyperparameters
│
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation (this file)
└── .gitignore                # Git ignore file
```

---

## Dataset

The dataset consists of images categorized into four classes:
1. **Monkeypox**
2. **Chickenpox**
3. **Measles**
4. **Normal** (healthy skin)

Images are stored in the `src/data/raw/` directory before processing.

### Dataset Preprocessing:
- **Resizing**: All images are resized to a uniform shape (e.g., 224x224).
- **Augmentation**: The following augmentation techniques are applied:
  - **Rotation Range**: 45 degrees
  - **Horizontal & Vertical Flips**
  - **Zoom Range**: 0.8 to 1.25
  - **Shear Range**: 45 degrees
  - **Shift Range**: 30% for height and width
  - **Brightness and Contrast Adjustments**

These augmentations help prevent overfitting and improve model generalization.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pawannitt-26/monkeypox_detection.git
cd monkeypox_detection
```

### 2. Install Dependencies

Make sure you have Python 3.7+ installed. You can install the necessary packages by running:

```bash
pip install -r requirements.txt
```

### 3. Set Up the Dataset

- Place your dataset images in the `src/data/raw/` folder.
- Use the provided `data_preprocessing.py` script or Jupyter notebook to preprocess the data and perform augmentation.
- The processed data will be split into `train/`, `val/`, and `test/` directories.

---

## Data Preprocessing & Augmentation

To preprocess and augment the dataset, run the following script:

### Using Python Script
```bash
python data_preprocessing.py
```

This will preprocess the data, apply augmentation, and save the output in the `src/data/train/`, `src/data/val/`, and `src/data/test/` directories.

### Using Jupyter Notebook
- Open the `data_preprocessing.ipynb`.
- Follow the steps in the notebook to preprocess and augment the dataset.

---

## Training the Model

Once the data is prepared, you can train the models using either deep learning or machine learning approaches.

### Training Deep Learning Models

#### 1. Using Jupyter Notebook
- Open `notebooks/deep_learning_model`.
- Define the CNN model architecture.
- Compile and train the model on the dataset.

### Machine Learning Models
You can experiment with the following models using the `machine_learning_model.ipynb`:
1. **K-Nearest Neighbors (K-NN)**
2. **Support Vector Machine (SVM)**
3. **XGBoost**
4. **Random Forest**

Run the notebook and evaluate these models on the same dataset splits.

---

## Model Evaluation

Once the models are trained, they are evaluated using various performance metrics:

- **Accuracy**
- **Confusion Matrix**
- **Precision, Recall, F1-Score**

### Visualization
Evaluation results, including confusion matrices and accuracy plots, are saved in the `results/plots/` directory.

---

## Results

After training and evaluation, you can find:
- **Trained Models** in the `results/models/` directory.
- **Evaluation Metrics** and **Visualizations** (e.g., confusion matrix, accuracy curves) in the `results/plots/` directory.

---

## Contributing

Contributions are welcome! If you find any issues or want to improve the project:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a Pull Request.

---

## Acknowledgements

- **TensorFlow/Keras** for the deep learning framework.
- **scikit-learn** for evaluation metrics.
- **XGBoost** for gradient boosting.
- **Matplotlib** and **Seaborn** for visualizations.

---

This structure ensures that the project is easy to follow, modular, and has proper documentation for different models.
