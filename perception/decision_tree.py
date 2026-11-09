
import numpy as np


class DecisionTree:

    def __init__(
        self,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_thresholds=20
    ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds

        self.tree = None

    def gini(self, labels):

        if len(labels) == 0:
            return 0

        _, counts = np.unique(
            labels,
            return_counts=True
        )

        probabilities = counts / len(labels)

        return 1 - np.sum(probabilities ** 2)

    def best_split(self, X, y):

        best_gain = 0
        best_feature = None
        best_threshold = None

        parent_impurity = self.gini(y)

        n_samples, n_features = X.shape

        for feature in range(n_features):

            values = np.unique(X[:, feature])

            if len(values) < 2:
                continue

            # Limit candidate thresholds for efficiency
            if len(values) > self.max_thresholds + 1:

                thresholds = np.quantile(
                    values,
                    np.linspace(
                        0,
                        1,
                        self.max_thresholds + 2
                    )[1:-1]
                )

                thresholds = np.unique(thresholds)

            else:

                thresholds = (
                    values[:-1] + values[1:]
                ) / 2

            for threshold in thresholds:

                left_mask = (
                    X[:, feature] <= threshold
                )

                right_mask = ~left_mask

                left_count = np.sum(left_mask)
                right_count = np.sum(right_mask)

                if (
                    left_count < self.min_samples_leaf
                    or right_count < self.min_samples_leaf
                ):
                    continue

                left_impurity = self.gini(y[left_mask])
                right_impurity = self.gini(y[right_mask])

                weighted_impurity = (
                    left_count / n_samples * left_impurity
                    + right_count / n_samples * right_impurity
                )

                gain = parent_impurity - weighted_impurity

                if gain > best_gain:

                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def majority_class(self, y):

        labels, counts = np.unique(
            y,
            return_counts=True
        )

        return labels[np.argmax(counts)]

    def build_tree(self, X, y, depth):

        prediction = self.majority_class(y)

        # Stopping conditions
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):

            return {
                "prediction": prediction
            }

        feature, threshold, gain = self.best_split(X, y)

        if feature is None or gain <= 0:

            return {
                "prediction": prediction
            }

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {

            "feature": feature,
            "threshold": float(threshold),
            "prediction": prediction,

            "left": self.build_tree(
                X[left_mask],
                y[left_mask],
                depth + 1
            ),

            "right": self.build_tree(
                X[right_mask],
                y[right_mask],
                depth + 1
            )
        }

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.tree = self.build_tree(
            X,
            y,
            0
        )

    def predict_one(self, sample, node):

        if "feature" not in node:

            return node["prediction"]

        feature = node["feature"]
        threshold = node["threshold"]

        if sample[feature] <= threshold:

            return self.predict_one(
                sample,
                node["left"]
            )

        return self.predict_one(
            sample,
            node["right"]
        )

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        return np.array([
            self.predict_one(sample, self.tree)
            for sample in X
        ])
