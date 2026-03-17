import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from src.lts import LearningTimeSeriesShapelets
from src.utils import load_ucr_dataset, normalize_with_sklearn


# -----------------------------
# Config
# -----------------------------
SEED = 42
np.random.seed(SEED)

PAPER_CFG = dict(
    K=0.3,
    L_min=0.2,
    R=3,
    lambda_w=0.01,
    learning_rate=0.01,
    max_iter=500,     # paper setting (adjust if too slow)
    alpha=-100
)

DATASET = dict(
    name="ItalyPowerDemand",
    train_path="data/ItalyPowerDemand_TRAIN",
    test_path="data/ItalyPowerDemand_TEST",
)

OUTDIR = os.path.join("results", DATASET["name"])
FIGDIR = os.path.join(OUTDIR, "figures")
TABDIR = os.path.join(OUTDIR, "tables")
LOGDIR = os.path.join(OUTDIR, "logs")

for d in [OUTDIR, FIGDIR, TABDIR, LOGDIR]:
    os.makedirs(d, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def save_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=np.unique(y_true),
        yticklabels=np.unique(y_true)
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_class_distribution(y, filename, title="Class distribution (train)"):
    counts = pd.Series(y).value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def save_series_by_class(X, y, filename, n_per_class=10, title="Time series by class (train)"):
    plt.figure(figsize=(10, 5))
    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0][:n_per_class]
        for i in idx:
            plt.plot(X[i], alpha=0.25)
        plt.plot(X[y == c].mean(axis=0), linewidth=3, label=f"Mean class {c}")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_accuracy_vs_train_fraction(fractions, accs, filename):
    plt.figure(figsize=(6, 4))
    plt.plot(fractions, accs, marker="o")
    plt.title("Accuracy vs Training Size (Paper hyperparameters)")
    plt.xlabel("Fraction of training data")
    plt.ylabel("Test accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_loss_curve(loss_history, filename):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.title("Training Loss Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_top_shapelets(lts_model, filename, top_n=5):
    # Importance across classes: sum_c |W|
    if lts_model.W.ndim == 3:
        importance = np.sum(np.abs(lts_model.W), axis=0)  # (R, K)
    else:
        importance = np.abs(lts_model.W)

    flat = importance.flatten()
    top_idx = np.argsort(flat)[-top_n:][::-1]

    fig, axes = plt.subplots(1, len(top_idx), figsize=(4 * len(top_idx), 4))
    if len(top_idx) == 1:
        axes = [axes]

    for ax, idx in zip(axes, top_idx):
        r, k = np.unravel_index(idx, importance.shape)
        L_r = (r + 1) * lts_model.L_min_len
        sh = lts_model.S[r, k, :L_r]
        sh = (sh - sh.mean()) / (sh.std() + 1e-8)

        ax.plot(sh, linewidth=3)
        ax.set_title(f"Scale {r+1}\nImp={flat[idx]:.3f}")
        ax.grid(alpha=0.3)

    plt.suptitle("Top Learned Shapelets (Paper setting)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# -----------------------------
# Load data
# -----------------------------
X_train, y_train, X_test, y_test = load_ucr_dataset(DATASET["train_path"], DATASET["test_path"])
X_train_scaled, X_test_scaled = normalize_with_sklearn(X_train, X_test)

# Save basic EDA figures
save_class_distribution(y_train, os.path.join(FIGDIR, "class_distribution.png"))
save_series_by_class(X_train_scaled, y_train, os.path.join(FIGDIR, "series_by_class.png"))

# -----------------------------
# Run methods
# -----------------------------
results = []

# ---- Baseline: 1-NN ----
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_scaled, y_train)
pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, pred_knn)
results.append({"method": "1NN", "accuracy": acc_knn})

save_confusion(y_test, pred_knn, "Confusion Matrix - 1NN", os.path.join(FIGDIR, "cm_1nn.png"))

# ---- Baseline: SVM (RBF) ----
svm = SVC()
svm.fit(X_train_scaled, y_train)
pred_svm = svm.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, pred_svm)
results.append({"method": "SVM (RBF)", "accuracy": acc_svm})

save_confusion(y_test, pred_svm, "Confusion Matrix - SVM", os.path.join(FIGDIR, "cm_svm.png"))

# ---- LTS (paper setting) ----
lts = LearningTimeSeriesShapelets(**PAPER_CFG)
lts.fit(X_train_scaled, y_train)
pred_lts = lts.predict(X_test_scaled)
acc_lts = accuracy_score(y_test, pred_lts)
results.append({"method": "LTS (paper)", "accuracy": acc_lts})

save_confusion(y_test, pred_lts, "Confusion Matrix - LTS", os.path.join(FIGDIR, "cm_lts.png"))

# If you added loss_history in fit()
if hasattr(lts, "loss_history") and lts.loss_history is not None:
    plot_loss_curve(lts.loss_history, os.path.join(FIGDIR, "loss_curve.png"))

plot_top_shapelets(lts, os.path.join(FIGDIR, "top_shapelets.png"), top_n=5)

# ---- LTS transform + Logistic Regression (extra baseline) ----
Xtr_lts = lts.transform(X_train_scaled)
Xte_lts = lts.transform(X_test_scaled)

# Handle potential NaNs (shouldn't happen for fixed-length dataset, but safe)
Xtr_lts = np.nan_to_num(Xtr_lts, nan=np.nanmean(Xtr_lts))
Xte_lts = np.nan_to_num(Xte_lts, nan=np.nanmean(Xte_lts))

logreg = LogisticRegression(max_iter=200)
logreg.fit(Xtr_lts, y_train)
pred_lr = logreg.predict(Xte_lts)
acc_lr = accuracy_score(y_test, pred_lr)
results.append({"method": "LTS-transform + LogReg", "accuracy": acc_lr})

save_confusion(y_test, pred_lr, "Confusion Matrix - LTS-transform + LogReg", os.path.join(FIGDIR, "cm_lts_logreg.png"))

# -----------------------------
# Training size study (paper setting)
# -----------------------------
fractions = [0.1, 0.3]
accs = []

for frac in fractions:
    n = int(frac * len(X_train_scaled))
    lts_tmp = LearningTimeSeriesShapelets(**PAPER_CFG)
    lts_tmp.fit(X_train_scaled[:n], y_train[:n])
    accs.append(accuracy_score(y_test, lts_tmp.predict(X_test_scaled)))

plot_accuracy_vs_train_fraction(fractions, accs, os.path.join(FIGDIR, "accuracy_vs_train_fraction.png"))

# -----------------------------
# Save results
# -----------------------------
results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
print("\n=== Benchmark results ===")
print(results_df)

results_df.to_csv(os.path.join(TABDIR, "benchmark_results.csv"), index=False)

with open(os.path.join(LOGDIR, "run_config.json"), "w", encoding="utf-8") as f:
    json.dump(
        {"seed": SEED, "dataset": DATASET, "paper_cfg": PAPER_CFG},
        f, indent=2
    )

print(f"\nSaved outputs to: {OUTDIR}")