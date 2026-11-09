
import numpy as np

from perception.decision_tree import DecisionTree


class RandomForest:

    def __init__(
        self,
        n_estimators=15,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_thresholds=24,
        random_state=42
    ):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds

        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        rng = np.random.default_rng(
            self.random_state
        )

        n_samples = len(X)

        self.trees = []

        for i in range(self.n_estimators):

            print(
                f"Training tree {i + 1}/{self.n_estimators}"
            )

            # Bootstrap sampling:
            # Randomly sample training observations with replacement
            bootstrap_indices = rng.choice(
                n_samples,
                size=n_samples,
                replace=True
            )

            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_thresholds=self.max_thresholds,
                max_features="sqrt",
                random_state=self.random_state + i + 1
            )

            tree.fit(
                X_bootstrap,
                y_bootstrap
            )

            self.trees.append(tree)

        return self

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        # Each tree independently predicts the class
        tree_predictions = np.array([
            tree.predict(X)
            for tree in self.trees
        ])

        predictions = []

        # Majority voting across trees
        for sample_predictions in tree_predictions.T:

            labels, counts = np.unique(
                sample_predictions,
                return_counts=True
            )

            predictions.append(
                labels[np.argmax(counts)]
            )

        return np.array(predictions)
