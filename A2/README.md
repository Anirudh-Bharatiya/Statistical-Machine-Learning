# MNIST Classification using PCA, FDA, and Discriminant Analysis

## Assignment Overview
This assignment involves building a classification pipeline to identify handwritten digits (0, 1, and 2) from the MNIST dataset. The assignment compares different dimensionality reduction techniques **Principal Component Analysis (PCA)** and **Fisher’s Discriminant Analysis (FDA)** and evaluates their impact on classification performance using **Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)**.

## Problem Statement
The goal is to classify a subset of the MNIST dataset containing only digits 0, 1, and 2. To simulate a constrained environment, the model is trained on a small, balanced subset:
*   **Training Set:** 300 samples (100 randomly selected samples per class).
*   **Test Set:** 300 samples (100 randomly selected samples per class).

## Task Description

### 1. Data Preprocessing
*   **Filtering:** Extract only classes 0, 1, and 2 from the raw MNIST dataset.
*   **Vectorization:** Convert `28 x 28` image matrices into feature vectors by **stacking columns**
*   **Normalization:** Rescale pixel values to a range of [0, 1].

### 2. Dimensionality Reduction using PCA
*   **Matrix Construction:** Form a data matrix X.
*   **Eigen-Decomposition:** Compute the covariance matrix and its eigenvectors/eigenvalues.
*   **Variance Retention:** 
    *   Project data while retaining **75%** of the variance.
    *   Project data while retaining **90%** of the variance.
    *   Project data onto the first **two principal components**.
*   **Reconstruction:** Reconstruct original images from the 75% variance projection and calculate the Mean Squared Error (MSE) for 5 sample images.

### 3. Class Projection using Fisher’s Discriminant Analysis (FDA)
*   **Scatter Matrices:** Compute the between-class scatter matrix and the within-class scatter matrix.
*   **Optimization:** Solve the generalized eigenvalue problem to find the optimal projection matrix that maximizes class separability.
*   **Projection:** Reduce the data to a low-dimensional space optimized for classification.

### 4. Classification & Evaluation
*   **Discriminant Analysis:** Implement LDA and QDA classifiers based on Maximum Likelihood Estimation (MLE).
*   **Performance Metrics:** Calculate classification accuracy for:
    *   FDA (2 components) + LDA/QDA.
    *   PCA (75% variance) + LDA.
    *   PCA (90% variance) + LDA.
    *   PCA (2 components) + LDA.
*   **Comparison:** Analyze how the choice of dimensionality reduction and the amount of retained variance affects the accuracy of the classifiers on both training and test sets.

## Deliverables
*   **Accuracy Reports:** Detailed classification accuracy for all combinations of reduction and classification methods.
*   **Visualizations:** 
    *   2D scatter plots of the transformed feature space for both PCA and FDA.
    *   Visual comparison of original images vs. PCA-reconstructed images.
*   **Analysis:** An evaluation of how PCA and FDA affect class separability and model generalization.