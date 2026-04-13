"""
Assignment 3 - Question 3

Regression decision stump and bagging on Fashion-MNIST classes 0, 1, and 2.
"""

from __future__ import annotations
import argparse
import urllib.request
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

SEED = 2023090
CLASS_LABELS = np.array([0, 1, 2], dtype=int)
N_BOOTSTRAPS = 5

@dataclass
class Stump:
    feature: int
    threshold: float
    left_value: float
    right_value: float

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)

def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n_images, n_rows, n_cols =[int.from_bytes(f.read(4), "big") for _ in range(4)]
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n_images, n_rows, n_cols)

def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n_labels =[int.from_bytes(f.read(4), "big") for _ in range(2)]
        return np.frombuffer(f.read(), dtype=np.uint8).astype(int)

def load_fashion_mnist(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir.mkdir(parents=True, exist_ok=True)
    urls =[
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]
    base_url = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
    
    paths =[]
    for fname in urls:
        target = data_dir / fname
        if not target.exists():
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(f"{base_url}/{fname}", target)
        paths.append(target)
        
    return _read_idx_images(paths[0]), _read_idx_labels(paths[1]), _read_idx_images(paths[2]), _read_idx_labels(paths[3])

def fit_pca(x_train: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    xc = x_train - mu
    cov = (xc.T @ xc) / (xc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    return mu, evecs[:, np.argsort(evals)[::-1][:p]]

def best_stump_feature_threshold(x: np.ndarray, y: np.ndarray) -> Tuple[int, float, float, float, float]:
    n, d = x.shape
    best_feature, best_threshold, best_ssr = 0, float(x[:, 0].mean()), np.inf
    best_left_value = best_right_value = float(y.mean())

    for j in range(d):
        order = np.argsort(x[:, j], kind="mergesort")
        xj, yj = x[order, j], y[order].astype(np.float64)
        if np.allclose(xj[0], xj[-1]): continue

        csum, csum_sq = np.cumsum(yj), np.cumsum(yj ** 2)

        for i in range(n - 1):
            if xj[i] == xj[i + 1]: continue
            n_left, n_right = i + 1, n - (i + 1)
            
            sum_left, sum_sq_left = csum[i], csum_sq[i]
            sum_right, sum_sq_right = csum[-1] - sum_left, csum_sq[-1] - sum_sq_left

            ssr = float((sum_sq_left - (sum_left ** 2) / n_left) + (sum_sq_right - (sum_right ** 2) / n_right))
            if ssr < best_ssr:
                best_ssr, best_feature = ssr, j
                best_threshold = float((xj[i] + xj[i + 1]) / 2.0)
                best_left_value, best_right_value = float(sum_left / n_left), float(sum_right / n_right)

    return best_feature, best_threshold, best_left_value, best_right_value, best_ssr

def fit_stump(x: np.ndarray, y: np.ndarray) -> Stump:
    f, t, l, r, _ = best_stump_feature_threshold(x, y)
    return Stump(f, t, l, r)

def predict_stump(stump: Stump, x: np.ndarray) -> np.ndarray:
    return np.where(x[:, stump.feature] <= stump.threshold, stump.left_value, stump.right_value).astype(np.float64)

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def oob_mse_for_stump(stump: Stump, x_full: np.ndarray, y_full: np.ndarray, idx: np.ndarray) -> float:
    in_bag = np.zeros(y_full.shape[0], dtype=bool)
    in_bag[np.unique(idx)] = True
    oob = ~in_bag
    if not np.any(oob): return 0.0
    return mse(y_full[oob], predict_stump(stump, x_full[oob]))

def run_experiment(data_dir: Path, output_dir: Path) -> None:
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_fashion_mnist(data_dir)
    mask_tr, mask_te = np.isin(y_train_raw, CLASS_LABELS), np.isin(y_test_raw, CLASS_LABELS)
    x_train, y_train = (x_train_raw[mask_tr].reshape(-1, 28*28) / 255.0).astype(np.float64), y_train_raw[mask_tr]
    x_test, y_test = (x_test_raw[mask_te].reshape(-1, 28*28) / 255.0).astype(np.float64), y_test_raw[mask_te]

    mu, components = fit_pca(x_train, 10)
    x_train_pca, x_test_pca = (x_train - mu) @ components, (x_test - mu) @ components

    y_train_num, y_test_num = y_train.astype(np.float64), y_test.astype(np.float64)

    stump = fit_stump(x_train_pca, y_train_num)
    single_train_mse = mse(y_train_num, predict_stump(stump, x_train_pca))
    single_test_mse = mse(y_test_num, predict_stump(stump, x_test_pca))

    rng = np.random.default_rng(SEED)
    boot_idx =[rng.integers(0, x_train_pca.shape[0], size=x_train_pca.shape[0]) for _ in range(N_BOOTSTRAPS)]

    stumps, oob_errors = [],[]
    for idx in boot_idx:
        x_b, y_b = x_train_pca[idx], y_train_num[idx]
        stump_b = fit_stump(x_b, y_b)
        stumps.append(stump_b)
        oob_errors.append(oob_mse_for_stump(stump_b, x_train_pca, y_train_num, idx))

    avg_oob_error = float(np.mean(oob_errors))
    bagged_train_pred = np.column_stack([predict_stump(s, x_train_pca) for s in stumps]).mean(axis=1)
    bagged_test_pred = np.column_stack([predict_stump(s, x_test_pca) for s in stumps]).mean(axis=1)
    bagged_test_mse = mse(y_test_num, bagged_test_pred)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    order = np.argsort(x_train_pca[:, stump.feature])
    plt.scatter(x_train_pca[order, stump.feature], y_train_num[order], s=8, alpha=0.15, label="True response")
    plt.plot(x_train_pca[order, stump.feature], predict_stump(stump, x_train_pca)[order], linewidth=2, label="Single stump")
    plt.plot(x_train_pca[order, stump.feature], bagged_train_pred[order], linewidth=2, label="Bagged stump")
    plt.xlabel("Chosen PCA feature value"); plt.ylabel("Response"); plt.legend(); plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "q3_prediction_comparison.png", dpi=200); plt.close()

    print("\nQuestion 3 results")
    print(f"Single stump train MSE: {single_train_mse:.6f}")
    print(f"Single stump test MSE: {single_test_mse:.6f}")
    print(f"Bagged train MSE: {mse(y_train_num, bagged_train_pred):.6f}")
    print(f"Bagged test MSE: {bagged_test_mse:.6f}")
    print(f"Average OOB MSE across 5 stumps: {avg_oob_error:.6f}")
    print(f"Chosen stump feature: {stump.feature}")
    print(f"Chosen stump threshold: {stump.threshold:.6f}")

if __name__ == "__main__":
    set_seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/fashion_mnist")
    parser.add_argument("--output-dir", type=str, default="outputs_q3")
    args = parser.parse_args()
    run_experiment(Path(args.data_dir), Path(args.output_dir))