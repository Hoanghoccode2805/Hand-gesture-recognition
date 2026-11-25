import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from preprocessing import Preprocessor

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.train_mlp import GestureMLP  

DATA_PATH = BASE_DIR / "data" / "gesture_data.csv"
RF_MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"
MLP_MODEL_PATH = BASE_DIR / "models" / "mlp.pth"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Open Palm", "Fist", "Peace", "Pointing"]


def evaluate_predictions(model_name, y_test, y_pred, elapsed_sec):
    ''' Calculate all evaluation metrics and prediction time '''
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    avg_ms = (elapsed_sec / len(y_test)) * 1000

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Inference (ms/sample)": avg_ms,
        "Confusion Matrix": cm,
    }


def predict_rf(model_path, X_test):
    """Load and make predictions using Random Forest (.joblib)."""
    model = joblib.load(model_path)
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - start
    return y_pred, elapsed


def predict_mlp(model_path, X_test):
    """Load and make predictions using PyTorch MLP (.pth)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureMLP(input_size=X_test.shape[1], num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(X_tensor)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    elapsed = time.perf_counter() - start
    return y_pred, elapsed


def save_confusion_matrices(results_list, save_path):
    """Plot and save the confusion matrices for the models side-by-side."""
    fig, axes = plt.subplots(1, len(results_list), figsize=(6 * len(results_list), 5))
    if len(results_list) == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        cm = res["Confusion Matrix"]
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{res['Model']} (Acc: {res['Accuracy']*100:.2f}%)")
        ax.set_xticks(np.arange(len(CLASS_NAMES)))
        ax.set_yticks(np.arange(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_ylabel("True label (True)")
        ax.set_xlabel("Predict (Predicted)")

        # Ghi con số trực tiếp vào từng ô
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion Matrix image saved at: {save_path}")


def main():

    _, X_test, _, y_test = Preprocessor.load_and_split_data()
    results = []

    # 1. Evaluating Random Forest
    if RF_MODEL_PATH.exists():
        y_pred_rf, time_rf = predict_rf(RF_MODEL_PATH, X_test)
        results.append(evaluate_predictions("Random Forest", y_test, y_pred_rf, time_rf))
    else:
        print(f"Not found {RF_MODEL_PATH}")

    # 2. Evaluating MLP (PyTorch)
    if MLP_MODEL_PATH.exists():
        y_pred_mlp, time_mlp = predict_mlp(MLP_MODEL_PATH, X_test)
        results.append(evaluate_predictions("MLP (PyTorch)", y_test, y_pred_mlp, time_mlp))
    else:
        print(f"Not found {MLP_MODEL_PATH}")

    if not results:
        print("No model to evaluate!")
        return

    # Print the overview comparison table
    summary_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "Confusion Matrix"} for r in results]
    )
    print("\n================ MODEL COMPARISON TABLE ================")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Print the Confusion Matrix in detail 
    for r in results:
        print(f"\n--- Confusion Matrix: {r['Model']} ---")
        cm_df = pd.DataFrame(r["Confusion Matrix"], index=CLASS_NAMES, columns=CLASS_NAMES)
        print(cm_df)

    # Save the CSV results table and images to the results/ folder.
    csv_out = RESULTS_DIR / "model_comparison.csv"
    summary_df.to_csv(csv_out, index=False)
    print(f"\n Comparison table saved at: {csv_out}")

    save_confusion_matrices(results, RESULTS_DIR / "confusion_matrices.png")


if __name__ == "__main__":
    main()