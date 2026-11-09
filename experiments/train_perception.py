
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from perception.knn import KNN
from perception.decision_tree import DecisionTree
from perception.evaluation import evaluate_model


def main():

    print("\nLoading LiDAR dataset...")

    dataset = pd.read_csv(
        "results/lidar_dataset.csv"
    )

    X = dataset.drop(
        columns=["label"]
    ).values

    y = dataset["label"].values

    labels = sorted(set(y))

    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {labels}")

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Scale features for distance-based KNN
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # 1. Our KNN implementation
    print("\nTraining KNN from scratch...")

    custom_knn = KNN(k=5)

    custom_knn.fit(
        X_train_scaled,
        y_train
    )

    custom_knn_predictions = custom_knn.predict(
        X_test_scaled
    )

    results.append(
        evaluate_model(
            "Custom KNN",
            y_test,
            custom_knn_predictions,
            labels
        )
    )

    # 2. Scikit-learn KNN
    print("\nTraining Scikit-learn KNN...")

    sklearn_knn = KNeighborsClassifier(
        n_neighbors=5
    )

    sklearn_knn.fit(
        X_train_scaled,
        y_train
    )

    sklearn_knn_predictions = sklearn_knn.predict(
        X_test_scaled
    )

    results.append(
        evaluate_model(
            "Sklearn KNN",
            y_test,
            sklearn_knn_predictions,
            labels
        )
    )

    # 3. Our Decision Tree implementation
    print("\nTraining Decision Tree from scratch...")

    custom_tree = DecisionTree(
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_thresholds=20
    )

    custom_tree.fit(
        X_train,
        y_train
    )

    custom_tree_predictions = custom_tree.predict(
        X_test
    )

    results.append(
        evaluate_model(
            "Custom Decision Tree",
            y_test,
            custom_tree_predictions,
            labels
        )
    )

    # 4. Scikit-learn Decision Tree
    print("\nTraining Scikit-learn Decision Tree...")

    sklearn_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    sklearn_tree.fit(
        X_train,
        y_train
    )

    sklearn_tree_predictions = sklearn_tree.predict(
        X_test
    )

    results.append(
        evaluate_model(
            "Sklearn Decision Tree",
            y_test,
            sklearn_tree_predictions,
            labels
        )
    )

    # Save comparison results
    os.makedirs("results", exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/perception_comparison.csv",
        index=False
    )

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))

    # Comparison graph
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]

    results_df.set_index("Model")[metrics].plot(
        kind="bar",
        figsize=(11, 6)
    )

    plt.title("LiDAR Classification Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        "results/perception_model_comparison.png"
    )

    plt.close()

    print("\nAll results saved in the results folder.")


if __name__ == "__main__":
    main()
