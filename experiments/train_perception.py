
import os
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from perception.knn import KNN
from perception.decision_tree import DecisionTree
from perception.random_forest import RandomForest
from perception.features import transform_dataset
from perception.evaluation import evaluate_model


def main():

    print("\nLoading LiDAR dataset...")

    dataset = pd.read_csv(
        "results/lidar_dataset.csv"
    )

    X_raw = dataset.drop(
        columns=["label"]
    ).values

    y = dataset["label"].values

    labels = sorted(set(y))

    print(f"Original dataset shape: {X_raw.shape}")
    print(f"Classes: {labels}")

    # Extract engineered features
    print("\nExtracting LiDAR features...")

    X = transform_dataset(X_raw)

    print(f"Engineered feature shape: {X.shape}")

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

    results = []

    # Scale features for KNN only
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Custom KNN
    print("\nTraining Custom KNN...")

    model = KNN(k=5)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    results.append(
        evaluate_model(
            "Custom KNN",
            y_test,
            predictions,
            labels
        )
    )

    # 2. Scikit-learn KNN
    print("\nTraining Scikit-learn KNN...")

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    results.append(
        evaluate_model(
            "Sklearn KNN",
            y_test,
            predictions,
            labels
        )
    )

    # 3. Custom Decision Tree
    print("\nTraining Custom Decision Tree...")

    model = DecisionTree(
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_thresholds=16,
        random_state=42
    )

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    print(
        f"Custom Decision Tree training time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    predictions = model.predict(X_test)

    results.append(
        evaluate_model(
            "Custom Decision Tree",
            y_test,
            predictions,
            labels
        )
    )

    # 4. Scikit-learn Decision Tree
    print("\nTraining Scikit-learn Decision Tree...")

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results.append(
        evaluate_model(
            "Sklearn Decision Tree",
            y_test,
            predictions,
            labels
        )
    )

    # 5. Custom Random Forest
    print("\nTraining Custom Random Forest...")

    model = RandomForest(
        n_estimators=12,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_thresholds=12,
        random_state=42
    )

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    print(
        f"Custom Random Forest training time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    predictions = model.predict(X_test)

    results.append(
        evaluate_model(
            "Custom Random Forest",
            y_test,
            predictions,
            labels
        )
    )

    # 6. Scikit-learn Random Forest
    print("\nTraining Scikit-learn Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results.append(
        evaluate_model(
            "Sklearn Random Forest",
            y_test,
            predictions,
            labels
        )
    )

    # Save comparison table
    os.makedirs("results", exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/perception_comparison.csv",
        index=False
    )

    print("\n\nFINAL MODEL COMPARISON")
    print(results_df.to_string(index=False))

    # Plot comparison
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]

    results_df.set_index("Model")[metrics].plot(
        kind="bar",
        figsize=(13, 7)
    )

    plt.title(
        "LiDAR Object Classification - Model Comparison"
    )

    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        "results/perception_model_comparison.png"
    )

    plt.close()

    print("\nComparison saved to results/perception_comparison.csv")
    print("Graphs saved in the results folder.")


if __name__ == "__main__":
    main()
