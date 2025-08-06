# DEEP-LEARNING-PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: MANISHA KUMARI

INTERN ID: CT12DN310

DOMAIN: DATA SCIENCE

DURATION: 12 WEEKS

MENTOR: NEELA SANTHOSH KUMAR

# IMAGE-CLASSIFICATION-CNN
CodTech Internship Task 2 – CIFAR-10 Image Classification using CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).  
The model is trained, evaluated, and visualized with accuracy/loss plots and confusion matrix.

## Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn (metrics)

## Workflow
### 1. Data Loading & Preprocessing
- Loads CIFAR-10 dataset from TensorFlow.
- Normalizes pixel values to range [0,1].
- Flattens labels.

### 2. Model Architecture
- **Conv2D + MaxPooling** layers for feature extraction.
- **Flatten + Dense layers** for classification.
- Output layer with softmax activation for 10 classes.

### 3. Training
- Optimizer: Adam  
- Loss: Sparse Categorical Crossentropy  
- Metrics: Accuracy  
- Epochs: 10  
- Validation on test dataset

### 4. Evaluation & Visualization
- Plots Accuracy vs Epochs and Loss vs Epochs.
- Generates Confusion Matrix.
- Displays sample predictions.

## How to Run
1. Install dependencies  
```bash
pip install tensorflow matplotlib seaborn scikit-learn
```
2. Run the script  
```bash
python cifar10_cnn.py
```
3. Outputs generated:
- Accuracy/Loss graphs
- Confusion matrix
- Saved model file: `cifar10_cnn_model.h5`

## Results
- **Test Accuracy**: ~70-72% (varies slightly per run)  
- Model shows clear improvement in accuracy and reduction in loss over epochs.

