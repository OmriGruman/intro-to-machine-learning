import numpy as np
import pandas as pd

from perceptron import Perceptron


def scale_features(X):
    print(f"Scaling features using their means & standard deviations...")

    # Calculating feature means and standard deviation
    means = X.mean(axis=0, keepdims=True)
    stds = X.std(axis=0, keepdims=True)

    # Standardize features
    return (X - means) / stds, means, stds


def train_test_split(X, y, test_size=0.2):
    assert 0 < test_size < 1

    print(f"Splitting data to train ({int(100 * (1 - test_size))}%) and test ({int(100 * test_size)}%)...")

    # All sample indices
    all_indices = np.arange(stop=len(y))
    test_indices = []

    # For each label, sample same portion of test samples
    for label in list(set(y)):

        # Calculate number of test samples from current label
        label_indices = all_indices[y == label]
        num_test_samples = int(len(label_indices) * test_size)

        # Randomly sample the indices
        label_test_indices = np.random.choice(label_indices, num_test_samples, replace=False)
        test_indices.extend(label_test_indices)

    # All other indices are train indices
    train_indices = np.delete(all_indices, test_indices)

    # Split data
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


if __name__ == '__main__':

    # Read breast cancer data
    data = pd.read_csv('Processed Wisconsin Diagnostic Breast Cancer.csv')
    X_all = data.drop(columns=['diagnosis']).to_numpy()
    y_all = data['diagnosis'].to_numpy()

    # Change 0 label to -1
    y_all[y_all == 0] = -1

    # Split data to train set and test set
    np.random.seed(0)
    X_train, y_train, X_test, y_test = train_test_split(X_all, y_all)

    # Scale features
    X_train, train_means, train_stds = scale_features(X_train)
    X_test = (X_test - train_means) / train_stds

    # Use perceptron
    p = Perceptron().fit(X_train, y_train)

    print(f"W: {p.weights}, B: {p.bias}")
    print(f"train score: {p.score(X_train, y_train):.4f}")
    print(f"train error: {p.error(X_train, y_train):.4f}")
    print(f"test score: {p.score(X_test, y_test):.4f}")
    print(f"test error: {p.error(X_test, y_test):.4f}")
