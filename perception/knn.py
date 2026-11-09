
import numpy as np


class KNN:

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):

        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        predictions = []

        for sample in X:

            # Euclidean distance from this sample
            # to every training observation
            distances = np.sqrt(
                np.sum(
                    (self.X_train - sample) ** 2,
                    axis=1
                )
            )

            nearest_indices = np.argsort(
                distances
            )[:self.k]

            nearest_labels = self.y_train[
                nearest_indices
            ]

            labels, counts = np.unique(
                nearest_labels,
                return_counts=True
            )

            predictions.append(
                labels[np.argmax(counts)]
            )

        return np.array(predictions)
