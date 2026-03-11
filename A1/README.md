# MNIST Classification using MLE and Discriminant Analysis

## Overview
This project implements a classification pipeline for handwritten digits (0,1,2) using:
- **Maximum Likelihood Estimation (MLE)** to estimate Gaussian parameters for each class.
- **Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)** for classification.
- t-SNE visualization of the feature space.

The dataset is the MNIST collection of 28x28 grayscale images. Only 100 samples per class are used for training and testing.

## Requirements
- Python 3.7+
- numpy
- matplotlib
- scikit-learn
- pandas (for fetch_openml)

Install dependencies with:
```bash
pip install numpy matplotlib scikit-learn pandas