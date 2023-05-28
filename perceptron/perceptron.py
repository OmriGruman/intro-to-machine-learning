import numpy as np


class Perceptron:
    def __init__(self):
        self.weights = None
        self.bias = 0

    def predict(self, X):
        assert self.weights is not None

        # Making predictions on samples
        return np.sign(X @ self.weights.reshape(-1, 1) + self.bias)

    def score(self, X, y):
        # Calculating model accuracy
        return np.sum(self.predict(X).flatten() == y) / len(y)

    def error(self, X, y):
        # Calculating model error
        return 1.0 - self.score(X, y)

    def fit(self, X, y, num_reps=-1):
        # Initialize weights according to number of samples
        self.weights = np.zeros(X.shape[1])

        print(f"Fitting model on {len(X)} samples...")

        # While errors exist AND did not reach maximum number of epochs
        errors_exist = True
        while errors_exist and num_reps != 0:
            errors_exist = False

            # Looking for prediction errors
            for x_, y_ in zip(X, y):

                # Predicted label is different from ground truth
                if self.predict(x_) != y_:
                    errors_exist = True

                    # Update weights & bias
                    self.weights += x_ * y_
                    self.bias += y_

        return self
