import numpy as np

from perceptron import Perceptron

if __name__ == '__main__':

    # Construct data
    X = np.array([
        [-2, -1],
        [0, 0],
        [2, 1],
        [1, 2],
        [-2, 2],
        [-3, 0]
    ])
    y = [-1, 1, 1, 1, -1, -1]

    # Use perceptron
    p = Perceptron().fit(X, y)

    print(f"W: {p.weights}, B: {p.bias}")
    print(f"predictions: {p.predict(X)}")
    print(f"score: {p.score(X, y)}")
    print(f"error: {p.error(X, y)}")
