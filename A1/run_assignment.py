import numpy as np
from sklearn.datasets import fetch_openml
from compute_estimates import compute_mle_estimates
from lda_qda import classify, calculate_accuracy
from sklearn.manifold import TSNE

def main():
    np.random.seed(42)

    print("Fetching MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X_raw, y_raw = mnist.data.astype('float32'), mnist.target.astype(int)

    # MNIST has 70,000 images. Standard split is 60k Train, 10k Test.
    X_train_full = X_raw[:60000]
    y_train_full = y_raw[:60000]
    X_test_full = X_raw[60000:]
    y_test_full = y_raw[60000:]

    def process_data(X_input, y_input):
        # Filter Classes 0, 1, 2
        mask = np.isin(y_input, [0,1,2])
        X_filtered = X_input[mask]
        y_filtered = y_input[mask]

        # Randomly sample 100 per class
        indices = []
        for c in [0,1,2]:
            class_indices = np.where(y_filtered == c)[0]
            indices.extend(np.random.choice(class_indices, 100, replace=False))

        X_sampled = X_filtered[indices]
        y_sampled = y_filtered[indices]

        # Preprocessing: Stack Columns and Normalize
        X_images = X_sampled.reshape(-1,28,28)
        X_col_stacked = X_images.transpose(0,2,1).reshape(-1,784)
        X_norm = X_col_stacked / 255.0
        return X_norm, y_sampled

    # Generate the final datasets
    print("Sampling and Preprocessing Data...")
    X_train, y_train = process_data(X_train_full, y_train_full)
    X_test, y_test = process_data(X_test_full, y_test_full)

    print("Train set size:", X_train.shape)
    print("Test set size:", X_test.shape)

    # MLE estimates
    mle_params = compute_mle_estimates(X_train, y_train, [0,1,2])

    # Classification
    lda_preds, lda_scores = classify(X_test, mle_params, method='LDA')
    qda_preds, qda_scores = classify(X_test, mle_params, method='QDA')

    # Accuracies
    print(f"LDA Test Accuracy: {calculate_accuracy(y_test, lda_preds)*100:.2f}%")
    print(f"QDA Test Accuracy: {calculate_accuracy(y_test, qda_preds)*100:.2f}%")

    # Discriminant values for a sample test point
    idx = 0
    print(f"\nSample Index {idx} (True Label: {y_test[idx]})")
    print(f"LDA Discriminant Values: {lda_scores[idx]}")
    print(f"QDA Discriminant Values: {qda_scores[idx]}")

    # Visualization (t-SNE)
    def save_tsne(X_vis, y_vis, title, name):
        print(f"Generating {name}...")
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X_vis)
        
        plt.figure(figsize=(10, 7))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Standard tableau colors
        for i, c in enumerate([0, 1, 2]):
            mask = (y_vis == c)
            plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                        c=colors[i], label=f'Digit {c}', alpha=0.7)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{name}.png")
        plt.close() 
        print(f"Visualization saved as {name}.png")

    save_tsne(X_train, y_train, "t-SNE Train Set (Digits 0,1,2)", "tsne_train")
    save_tsne(X_test, y_test, "t-SNE Test Set (Digits 0,1,2)", "tsne_test")

if __name__ == "__main__":
    main()