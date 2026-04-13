"""
Assignment 3 - Question 2

Decision tree, bagging, and random forest classification on MNIST
using PCA-reduced features from the three classes 0, 1, and 2.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np

SEED = 2023090
CLASS_LABELS = np.array([0, 1, 2], dtype=int)
N_BOOTSTRAPS = 5

@dataclass
class TreeNode:
    is_leaf: bool
    prediction: int
    class_counts: np.ndarray
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)

def load_mnist_npz(dataset_path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset_path is None: dataset_path = Path("data") / "mnist.npz"
    if not dataset_path.exists():
        import urllib.request
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", dataset_path)
    data = np.load(dataset_path)
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]

def preprocess_mnist(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    if x.ndim == 3: x = x.reshape(x.shape[0], -1)
    return x / 255.0

def select_classes(x: np.ndarray, y: np.ndarray, classes: np.ndarray = CLASS_LABELS) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isin(y, classes)
    return x[mask], y[mask]

def fit_pca(x_train: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    xc = x_train - mu
    cov = (xc.T @ xc) / (xc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    return mu, evecs[:, np.argsort(evals)[::-1][:p]]

def apply_pca(x: np.ndarray, mu: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (x - mu) @ components

def class_counts(y: np.ndarray, classes: np.ndarray = CLASS_LABELS) -> np.ndarray:
    return np.array([(y == c).sum() for c in classes], dtype=int)

def majority_class(y: np.ndarray, classes: np.ndarray = CLASS_LABELS) -> int:
    return int(classes[0]) if y.size == 0 else int(classes[np.argmax(class_counts(y, classes))])

def gini(y: np.ndarray, classes: np.ndarray = CLASS_LABELS) -> float:
    if y.size == 0: return 0.0
    probs = class_counts(y, classes).astype(np.float64) / y.size
    return float(1.0 - np.sum(probs ** 2))

def weighted_gini(y_left: np.ndarray, y_right: np.ndarray, classes: np.ndarray = CLASS_LABELS) -> float:
    n = y_left.size + y_right.size
    if n == 0: return 0.0
    return (y_left.size / n) * gini(y_left, classes) + (y_right.size / n) * gini(y_right, classes)

def candidate_threshold(x_col: np.ndarray) -> float:
    return float(np.median(x_col))

def best_split(x: np.ndarray, y: np.ndarray, feature_indices: Sequence[int], classes: np.ndarray = CLASS_LABELS) -> Optional[Tuple[int, float, np.ndarray, np.ndarray, float]]:
    best, best_value = None, np.inf
    for j in feature_indices:
        thr = candidate_threshold(x[:, j])
        left_mask = x[:, j] <= thr
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0: continue
        value = weighted_gini(y[left_mask], y[right_mask], classes)
        if value < best_value: best, best_value = (int(j), float(thr), left_mask, right_mask, float(value)), value
    return best

def build_three_leaf_tree(x: np.ndarray, y: np.ndarray, feature_sampler: Callable[[int], np.ndarray], rng: np.random.Generator, classes: np.ndarray = CLASS_LABELS) -> TreeNode:
    n_features = x.shape[1]
    root_split = best_split(x, y, feature_sampler(n_features), classes)

    if root_split is None:
        return TreeNode(True, majority_class(y, classes), class_counts(y, classes))

    root_feature, root_thr, left_mask, right_mask, _ = root_split
    x_left, y_left, x_right, y_right = x[left_mask], y[left_mask], x[right_mask], y[right_mask]

    left_leaf = TreeNode(True, majority_class(y_left, classes), class_counts(y_left, classes))
    right_leaf = TreeNode(True, majority_class(y_right, classes), class_counts(y_right, classes))
    root = TreeNode(False, majority_class(y, classes), class_counts(y, classes), feature=root_feature, threshold=root_thr, left=left_leaf, right=right_leaf)

    candidates =[]
    for side_name, x_child, y_child in[("left", x_left, y_left), ("right", x_right, y_right)]:
        if y_child.size <= 1: continue
        child_split = best_split(x_child, y_child, feature_sampler(n_features), classes)
        if child_split is None: continue
        feat, thr, c_left_mask, c_right_mask, _ = child_split
        y_c_left, y_c_right = y_child[c_left_mask], y_child[c_right_mask]
        
        y_other = y_right if side_name == "left" else y_left
        total_n = y.size
        global_value = ((y_c_left.size / total_n) * gini(y_c_left, classes) + 
                        (y_c_right.size / total_n) * gini(y_c_right, classes) + 
                        (y_other.size / total_n) * gini(y_other, classes))
        candidates.append((global_value, side_name, feat, thr, c_left_mask, c_right_mask))

    if not candidates: return root

    candidates.sort(key=lambda item: item[0])
    _, side_name, feat2, thr2, c_left_mask, c_right_mask = candidates[0]

    split_parent_y = y_left if side_name == "left" else y_right
    other_leaf = right_leaf if side_name == "left" else left_leaf

    left_child = TreeNode(True, majority_class(split_parent_y[c_left_mask], classes), class_counts(split_parent_y[c_left_mask], classes))
    right_child = TreeNode(True, majority_class(split_parent_y[c_right_mask], classes), class_counts(split_parent_y[c_right_mask], classes))
    
    internal = TreeNode(False, majority_class(split_parent_y, classes), class_counts(split_parent_y, classes), feature=feat2, threshold=thr2, left=left_child, right=right_child)

    if side_name == "left":
        root.left, root.right = internal, other_leaf
    else:
        root.right, root.left = internal, other_leaf

    return root

def predict_one(node: TreeNode, x: np.ndarray) -> int:
    if node.is_leaf or node.feature is None or node.threshold is None: return int(node.prediction)
    return predict_one(node.left, x) if x[node.feature] <= node.threshold and node.left else \
           predict_one(node.right, x) if node.right else int(node.prediction)

def predict_tree(node: TreeNode, x: np.ndarray) -> np.ndarray:
    return np.array([predict_one(node, row) for row in x], dtype=int)

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))

def class_wise_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray = CLASS_LABELS) -> Dict[int, float]:
    out = {}
    for c in classes:
        mask = y_true == c
        out[int(c)] = float(np.mean(y_pred[mask] == y_true[mask])) if mask.any() else 0.0
    return out

def oob_error_for_tree(tree: TreeNode, x_full: np.ndarray, y_full: np.ndarray, bootstrap_idx: np.ndarray) -> float:
    in_bag = np.zeros(y_full.shape[0], dtype=bool)
    in_bag[np.unique(bootstrap_idx)] = True
    oob = ~in_bag
    return float(np.mean(predict_tree(tree, x_full[oob]) != y_full[oob])) if np.any(oob) else 0.0

def fit_classification_tree(x: np.ndarray, y: np.ndarray, k_features: Optional[int], rng: np.random.Generator) -> TreeNode:
    sampler = (lambda p: np.arange(p)) if k_features is None or k_features >= x.shape[1] else (lambda p: rng.choice(p, size=k_features, replace=False))
    return build_three_leaf_tree(x, y, sampler, rng)

def evaluate_ensemble(trees: List[TreeNode], x: np.ndarray) -> np.ndarray:
    preds = np.column_stack([predict_tree(t, x) for t in trees])
    return np.array([np.unique(row)[np.argmax(np.unique(row, return_counts=True)[1])] for row in preds], dtype=int)

def run_experiment(mnist_path: Path | None, output_dir: Path) -> Dict[str, float]:
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_mnist_npz(mnist_path)
    x_train_raw, y_train = select_classes(x_train_raw, y_train_raw)
    x_test_raw, y_test = select_classes(x_test_raw, y_test_raw)
    
    x_train_pca = apply_pca(preprocess_mnist(x_train_raw), *fit_pca(preprocess_mnist(x_train_raw), 10))
    x_test_pca = apply_pca(preprocess_mnist(x_test_raw), *fit_pca(preprocess_mnist(x_train_raw), 10))

    rng = np.random.default_rng(SEED)
    bootstrap_indices =[rng.integers(0, x_train_pca.shape[0], size=x_train_pca.shape[0]) for _ in range(N_BOOTSTRAPS)]

    base_trees, base_oob = [],[]
    print("Training plain decision trees on bootstrap samples...")
    for idx in bootstrap_indices:
        tree = fit_classification_tree(x_train_pca[idx], y_train[idx], k_features=None, rng=rng)
        base_trees.append(tree)
        base_oob.append(oob_error_for_tree(tree, x_train_pca, y_train, idx))

    base_oob_avg = float(np.mean(base_oob))
    base_test_pred = evaluate_ensemble(base_trees, x_test_pca)
    base_test_acc = accuracy(y_test, base_test_pred)

    single_tree = fit_classification_tree(x_train_pca, y_train, k_features=None, rng=rng)
    single_test_pred = predict_tree(single_tree, x_test_pca)
    single_test_acc, single_test_class_acc = accuracy(y_test, single_test_pred), class_wise_accuracy(y_test, single_test_pred)

    candidate_ks, rf_oob_curve = list(range(1, x_train_pca.shape[1] + 1)),[]
    print("Tuning random forest k using OOB error...")
    
    for k in candidate_ks:
        k_oob =[]
        local_rng = np.random.default_rng(SEED + k)  # FIX: initialized OUTSIDE the inner loop
        for idx in bootstrap_indices:
            tree = fit_classification_tree(x_train_pca[idx], y_train[idx], k_features=k, rng=local_rng)
            k_oob.append(oob_error_for_tree(tree, x_train_pca, y_train, idx))
        rf_oob_curve.append(float(np.mean(k_oob)))

    best_k = int(candidate_ks[np.argmin(rf_oob_curve)])
    rf_trees, rf_oob = [],[]
    final_rng = np.random.default_rng(SEED + 999)    # FIX: initialized OUTSIDE the inner loop
    
    for idx in bootstrap_indices:
        tree = fit_classification_tree(x_train_pca[idx], y_train[idx], k_features=best_k, rng=final_rng)
        rf_trees.append(tree)
        rf_oob.append(oob_error_for_tree(tree, x_train_pca, y_train, idx))

    rf_oob_avg = float(np.mean(rf_oob))
    rf_test_pred = evaluate_ensemble(rf_trees, x_test_pca)
    rf_test_acc = accuracy(y_test, rf_test_pred)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5)); plt.plot(candidate_ks, rf_oob_curve, marker="o")
    plt.xlabel("k features"); plt.ylabel("Avg OOB error"); plt.title("RF tuning"); plt.grid(True, alpha=0.25)
    plt.tight_layout(); plt.savefig(output_dir / "q2_rf_k_tuning.png", dpi=200); plt.close()

    print("\nQuestion 2 results")
    print(f"Single-tree test accuracy: {single_test_acc*100:.2f}%\nBase tree ensemble test accuracy: {base_test_acc*100:.2f}%\nBase tree average OOB error: {base_oob_avg:.6f}")
    print(f"Chosen k for RF: {best_k}\nRandom forest test accuracy: {rf_test_acc*100:.2f}%\nRandom forest average OOB error: {rf_oob_avg:.6f}")
    print("\nClass-wise accuracy (single tree):", single_test_class_acc)
    print("Class-wise accuracy (bagging):", class_wise_accuracy(y_test, base_test_pred))
    print("Class-wise accuracy (random forest):", class_wise_accuracy(y_test, rf_test_pred))

    return {"single_tree_test_accuracy": single_test_acc, "bagging_test_accuracy": base_test_acc, "bagging_oob_error": base_oob_avg, "best_k": float(best_k), "rf_test_accuracy": rf_test_acc, "rf_oob_error": rf_oob_avg}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs_q2")
    return parser.parse_args()

if __name__ == "__main__":
    set_seed(SEED)
    args = parse_args()
    run_experiment(Path(args.mnist_path) if args.mnist_path else None, Path(args.output_dir))