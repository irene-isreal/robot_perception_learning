
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


def evaluate_model(
    name,
    y_true,
    y_pred,
    labels,
    save_results=True
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"\n{'=' * 50}")
    print(name)
    print(f"{'=' * 50}")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0
        )
    )

    if save_results:

        os.makedirs("results", exist_ok=True)

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=labels
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=labels
        )

        display.plot(
            cmap="Blues",
            xticks_rotation=30
        )

        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()

        filename = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        plt.savefig(
            f"results/{filename}_confusion_matrix.png"
        )

        plt.close()

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }
