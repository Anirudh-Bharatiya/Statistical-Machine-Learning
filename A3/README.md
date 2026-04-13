# SML Assignment 3 — Roll No. 2023090

## Overview

This submission is split into three standalone Python files:

- `2023090_A3_Q1.py`
- `2023090_A3_Q2.py`
- `2023090_A3_Q3.py`

Each script is self-contained, uses the seed `2023090`, and saves figures and metrics in its own output folder.

## Question 1: Ridge and Lasso Classification after PCA

### Data preparation
- Load the MNIST train and test splits from `mnist.npz`.
- Keep only classes `0`, `1`, and `2`.
- Flatten the images and normalize pixel values by dividing by `255`.

### PCA
PCA is fitted on the full training subset of the three classes and then used to project both train and test data.

### Target vectors
For each class $k \in \{0, 1, 2\}$:
$$y_k = 
\begin{cases} 
1 & \text{if the sample belongs to class } k \\ 
0 & \text{otherwise} 
\end{cases}$$

### Ridge Regression
The ridge model is fit in one-vs-rest form using the PCA features. The ridge objective is:
$$\frac{1}{n} \|y - Xw - b\|_2^2 + \lambda \|w\|_2^2$$

The intercept $b$ is not regularized.


### Lasso regression
Lasso is fit in one-vs-rest form using `sklearn.linear_model.Lasso`, which is the only scikit-learn component used in the assignment.

### Reported plots
- Training and test MSE vs. \(\lambda\) for ridge and lasso.
- Number of non-zero lasso coefficients vs. \(\lambda\).
- Regularization paths for ridge and lasso for class 1.
- Ridge training and test MSE vs. PCA dimension \(p\).

### Classification accuracy
The final test classification label is obtained by `argmax` over the three regression scores.

### Brief interpretation of plots
- For small values of $\lambda$, ridge and lasso both achieve low training and test MSE.
- As $\lambda$ becomes large, both methods underfit and the MSE increases.
- Lasso progressively sets more coefficients to zero, showing its sparsity property.
- Ridge coefficient paths shrink smoothly toward zero as $\lambda$ increases.
- Lasso coefficient paths become exactly zero at different regularization strengths.
- Increasing the PCA dimension generally improves model capacity, though the best test performance depends on the trade-off between bias and variance.

## Question 2: Decision Trees, Bagging, and Random Forest

### PCA
The same preprocessing is used, with PCA dimension fixed to `p = 10`.

### Decision tree construction
A greedy binary tree with three terminal nodes is built using Gini impurity.
For a node $S$:
$$Gini(S) = 1 - \sum_{k=1}^{K} p_k^2$$

For a split into children $S_i$:
$$Gini_{weighted} = \sum_i \frac{|S_i|}{|S|} Gini(S_i)$$

The implementation uses the median of each feature as the candidate threshold at each node, then chooses the feature/threshold pair with the minimum weighted Gini.

### Bagging
- Five bootstrap samples are created.
- A three-leaf tree is trained on each bootstrap sample.
- OOB error is computed using samples not selected in that bootstrap sample.
- Final test prediction uses majority vote across the five trees.

### Random forest
- The same bootstrap samples are reused.
- At each split, only `k` randomly selected features are considered.
- `k` is chosen by minimizing the average OOB error.

### Brief interpretation
- The single tree is interpretable but limited.
- Bagging stabilizes predictions by averaging multiple bootstrap trees.
- Random forest adds feature subsampling to reduce correlation among trees.
- In this submission, the best `k` is close to `p`, so the random forest behaves similarly to bagging.

## Question 3: Regression Stump and Bagging on Fashion-MNIST

### Data preparation
- Load Fashion-MNIST.
- Keep only classes `0`, `1`, and `2`.
- Flatten and normalize the images.
- Apply PCA with `p = 10`.

### Decision stump regression
The label values `0`, `1`, and `2` are treated as a numeric regression target. For each feature, all candidate thresholds are the midpoints between consecutive sorted values.
For a split into left and right regions:
$$SSR = \sum_{i \in L}(y_i - \bar{y}_L)^2 + \sum_{i \in R}(y_i - \bar{y}_R)^2$$

The split minimizing $SSR$ is selected.


### Bagging
- Five bootstrap samples are drawn.
- A regression stump is trained on each bootstrap sample.
- OOB MSE is computed from samples left out of each bootstrap sample.
- The bagged prediction is the average of the five stump predictions.

### Brief interpretation
- A single stump produces a very simple step-like prediction.
- Bagging averages several stumps trained on different bootstrap samples.
- This reduces variance and usually gives slightly lower test MSE than a single stump.

## Files and outputs

Each script saves:
- figures in `outputs_q1`, `outputs_q2`, and `outputs_q3`

## How to run

```bash
python 2023090_A3_Q1.py --output-dir outputs_q1
python 2023090_A3_Q2.py --output-dir outputs_q2
python 2023090_A3_Q3.py --data-dir data/fashion_mnist --output-dir outputs_q3