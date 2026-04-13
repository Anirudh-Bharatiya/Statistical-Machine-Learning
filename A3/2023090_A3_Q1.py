"""
Assignment 3 - Question 1

Ridge and Lasso regression classification on MNIST (classes 0, 1, 2)
after PCA reduction to a chosen dimension p.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

SEED = 2023090
LAMBDAS = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], dtype=float)
CLASS_LABELS = np.array([0, 1, 2], dtype=int)

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)

def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): return destination
    import urllib.request
    print(f"Downloading dataset from {url}")
    urllib.request.urlretrieve(url, destination)
    return destination

def load_mnist_npz(dataset_path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    if dataset_path is None: dataset_path = Path("data") / "mnist.npz"
    if not dataset_path.exists(): download_file(url, dataset_path)
    data = np.load(dataset_path)
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]

def preprocess_mnist(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    if x.ndim == 3: x = x.reshape(x.shape[0], -1)
    return x / 255.0

def select_classes(x: np.ndarray, y: np.ndarray, classes: np.ndarray = CLASS_LABELS) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isin(y, classes)
    return x[mask], y[mask]

def fit_pca(x_train: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    xc = x_train - mu
    cov = (xc.T @ xc) / (xc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    return mu, evecs[:, :p], evals

def apply_pca(x: np.ndarray, mu: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (x - mu) @ components

def one_hot_targets(y: np.ndarray, classes: np.ndarray = CLASS_LABELS) -> np.ndarray:
    targets = np.zeros((y.shape[0], len(classes)), dtype=np.float64)
    for idx, c in enumerate(classes): targets[:, idx] = (y == c).astype(np.float64)
    return targets

def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> Tuple[np.ndarray, float]:
    n, d = x.shape
    x_mean, y_mean = x.mean(axis=0), y.mean()
    xc, yc = x - x_mean, y - y_mean
    a = (xc.T @ xc) / n + lam * np.eye(d)
    b = (xc.T @ yc) / n
    w = np.linalg.solve(a, b)
    intercept = float(y_mean - float(x_mean @ w))
    return w, intercept

def ridge_fit_multitarget(x: np.ndarray, y_targets: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    ws, bs = [],[]
    for j in range(y_targets.shape[1]):
        w, b = ridge_fit(x, y_targets[:, j], lam)
        ws.append(w.reshape(-1))
        bs.append(b)
    return np.column_stack(ws), np.array(bs, dtype=np.float64)

def ridge_predict(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w + b

def lasso_fit_multitarget(x: np.ndarray, y_targets: np.ndarray, lam: float, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    ws, bs = [],[]
    for j in range(y_targets.shape[1]):
        model = Lasso(alpha=float(lam), fit_intercept=True, max_iter=20000, tol=1e-6, random_state=seed, selection="cyclic")
        model.fit(x, y_targets[:, j])
        ws.append(model.coef_.astype(np.float64))
        bs.append(float(model.intercept_))
    return np.column_stack(ws), np.array(bs, dtype=np.float64)

def mse_by_class(x: np.ndarray, y_targets: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean((ridge_predict(x, w, b) - y_targets) ** 2, axis=0)

def classification_accuracy(x: np.ndarray, y_true: np.ndarray, w: np.ndarray, b: np.ndarray, classes: np.ndarray = CLASS_LABELS) -> float:
    scores = ridge_predict(x, w, b)
    y_pred = classes[np.argmax(scores, axis=1)]
    return float(np.mean(y_pred == y_true))

def plot_error_curves(lambdas: np.ndarray, train_err: np.ndarray, test_err: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, train_err, marker="o", label="Training MSE")
    plt.plot(lambdas, test_err, marker="o", label="Test MSE")
    plt.xscale("log")
    plt.xlabel("λ")
    plt.ylabel("Average MSE")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_nonzero(lambdas: np.ndarray, nonzero: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, nonzero, marker="o")
    plt.xscale("log")
    plt.xlabel("λ")
    plt.ylabel("Number of non-zero coefficients")
    plt.title("Lasso sparsity pattern")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_regularization_path(lambdas: np.ndarray, coefs: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for j in range(coefs.shape[1]): plt.plot(lambdas, coefs[:, j], linewidth=1)
    plt.xscale("log")
    plt.xlabel("λ")
    plt.ylabel("Coefficient value")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_complexity(p_values: List[int], train_err: List[float], test_err: List[float], path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(p_values, train_err, marker="o", label="Training MSE")
    plt.plot(p_values, test_err, marker="o", label="Test MSE")
    plt.xlabel("Number of PCA dimensions p")
    plt.ylabel("Average MSE")
    plt.title("Ridge regression error vs model complexity")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def run_experiment(train_path: Path | None, p: int, output_dir: Path) -> Dict[str, float]:
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_mnist_npz(train_path)
    x_train_raw, y_train = select_classes(x_train_raw, y_train_raw)
    x_test_raw, y_test = select_classes(x_test_raw, y_test_raw)
    
    x_train, x_test = preprocess_mnist(x_train_raw), preprocess_mnist(x_test_raw)
    mu, components, _ = fit_pca(x_train, p)
    x_train_pca, x_test_pca = apply_pca(x_train, mu, components), apply_pca(x_test, mu, components)
    y_train_targets, y_test_targets = one_hot_targets(y_train), one_hot_targets(y_test)

    ridge_train_mse, ridge_test_mse, ridge_models = [], [],[]
    lasso_train_mse, lasso_test_mse, lasso_models, lasso_nonzero = [], [], [], []
    ridge_paths_class1, lasso_paths_class1 = [],[]

    print("Running ridge and lasso over lambda grid...")
    for lam in LAMBDAS:
        wr, br = ridge_fit_multitarget(x_train_pca, y_train_targets, float(lam))
        ridge_models.append((wr, br))
        ridge_train_mse.append(float(np.mean(mse_by_class(x_train_pca, y_train_targets, wr, br))))
        ridge_test_mse.append(float(np.mean(mse_by_class(x_test_pca, y_test_targets, wr, br))))

        wl, bl = lasso_fit_multitarget(x_train_pca, y_train_targets, float(lam), seed=SEED)
        lasso_models.append((wl, bl))
        lasso_train_mse.append(float(np.mean(mse_by_class(x_train_pca, y_train_targets, wl, bl))))
        lasso_test_mse.append(float(np.mean(mse_by_class(x_test_pca, y_test_targets, wl, bl))))
        lasso_nonzero.append(int(np.sum(np.abs(wl) > 1e-10)))

        ridge_paths_class1.append(wr[:, 1])
        lasso_paths_class1.append(wl[:, 1])

    ridge_best_idx, lasso_best_idx = int(np.argmin(ridge_test_mse)), int(np.argmin(lasso_test_mse))
    ridge_best_lam, lasso_best_lam = float(LAMBDAS[ridge_best_idx]), float(LAMBDAS[lasso_best_idx])
    ridge_best_w, ridge_best_b = ridge_models[ridge_best_idx]
    lasso_best_w, lasso_best_b = lasso_models[lasso_best_idx]

    ridge_best_acc = classification_accuracy(x_test_pca, y_test, ridge_best_w, ridge_best_b)
    lasso_best_acc = classification_accuracy(x_test_pca, y_test, lasso_best_w, lasso_best_b)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_error_curves(LAMBDAS, ridge_train_mse, ridge_test_mse, "Ridge regression classification: MSE vs λ", output_dir / "q1_ridge_mse.png")
    plot_error_curves(LAMBDAS, lasso_train_mse, lasso_test_mse, "Lasso regression classification: MSE vs λ", output_dir / "q1_lasso_mse.png")
    plot_nonzero(LAMBDAS, lasso_nonzero, output_dir / "q1_lasso_nonzero.png")
    plot_regularization_path(LAMBDAS, np.asarray(ridge_paths_class1), "Ridge regularization path for class 1", output_dir / "q1_ridge_path_class1.png")
    plot_regularization_path(LAMBDAS, np.asarray(lasso_paths_class1), "Lasso regularization path for class 1", output_dir / "q1_lasso_path_class1.png")

    p_values, ridge_complexity_train, ridge_complexity_test =[2, 5, 10, 20, 30], [],[]
    print("Running ridge regression for multiple PCA dimensions...")
    for p_i in p_values:
        mu_i, components_i, _ = fit_pca(x_train, p_i)
        x_train_i, x_test_i = apply_pca(x_train, mu_i, components_i), apply_pca(x_test, mu_i, components_i)
        w_i, b_i = ridge_fit_multitarget(x_train_i, y_train_targets, ridge_best_lam)
        ridge_complexity_train.append(float(np.mean(mse_by_class(x_train_i, y_train_targets, w_i, b_i))))
        ridge_complexity_test.append(float(np.mean(mse_by_class(x_test_i, y_test_targets, w_i, b_i))))

    plot_complexity(p_values, ridge_complexity_train, ridge_complexity_test, output_dir / "q1_ridge_complexity.png")

    summary = {
        "ridge_best_test_mse": float(ridge_test_mse[ridge_best_idx]),
        "ridge_best_test_accuracy": float(ridge_best_acc),
        "lasso_best_test_mse": float(lasso_test_mse[lasso_best_idx]),
        "lasso_best_test_accuracy": float(lasso_best_acc),
    }

    print("\nQuestion 1 results")
    print(f"Best ridge λ: {ridge_best_lam}\nBest ridge test MSE: {summary['ridge_best_test_mse']:.6f}\nBest ridge test accuracy: {summary['ridge_best_test_accuracy']*100:.2f}%")
    print(f"Best lasso λ: {lasso_best_lam}\nBest lasso test MSE: {summary['lasso_best_test_mse']:.6f}\nBest lasso test accuracy: {summary['lasso_best_test_accuracy']*100:.2f}%")

    return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 3 - Question 1")
    parser.add_argument("--mnist-path", type=str, default="", help="Path to mnist.npz (optional).")
    parser.add_argument("--p", type=int, default=10, help="PCA dimension for the main experiment.")
    parser.add_argument("--output-dir", type=str, default="outputs_q1", help="Directory for figures and metrics.")
    return parser.parse_args()

def main() -> None:
    set_seed(SEED)
    args = parse_args()
    mnist_path = Path(args.mnist_path) if args.mnist_path else None
    run_experiment(mnist_path, args.p, Path(args.output_dir))

if __name__ == "__main__":
    main()