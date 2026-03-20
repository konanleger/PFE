import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.lts import LearningTimeSeriesShapelets
from src.utils import load_ucr_dataset, normalize_with_sklearn


# =========================================================
# CONFIGURATION
# =========================================================
SEED = 42
np.random.seed(SEED)

PAPER_CFG = dict(
    K=0.15,
    L_min=0.125,
    R=2,
    lambda_w=0.01,
    learning_rate=0.01,
    max_iter=100,   
    alpha=-100
)

DATASET_NAME = "synthetic_control"
TRAIN_PATH = "data/synthetic_control_TRAIN"
TEST_PATH = "data/synthetic_control_TEST"

OUTDIR = os.path.join("results", DATASET_NAME)
FIGDIR = os.path.join(OUTDIR, "figures")
TABDIR = os.path.join(OUTDIR, "tables")
LOGDIR = os.path.join(OUTDIR, "logs")

for d in [OUTDIR, FIGDIR, TABDIR, LOGDIR]:
    os.makedirs(d, exist_ok=True)


# =========================================================
# FONCTIONS UTILITAIRES
# =========================================================
def save_confusion_matrix(y_true, y_pred, title, filepath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y_true),
        yticklabels=np.unique(y_true)
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def save_class_distribution(y, filepath):
    class_counts = pd.Series(y).value_counts().sort_index()

    print("Class distribution:")
    print(class_counts)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index.astype(str), y=class_counts.values)
    plt.title("Class Distribution (Training)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def save_series_by_class(X, y, filepath, n_per_class=10):
    plt.figure(figsize=(10, 5))

    for c in np.unique(y):
        idx = np.where(y == c)[0][:n_per_class]
        for i in idx:
            plt.plot(X[i], alpha=0.25)

        mean_series = X[y == c].mean(axis=0)
        plt.plot(mean_series, linewidth=3, label=f"Mean class {c}")

    plt.title("Time Series by Class (Train)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_loss_curve(loss_history, filepath):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, linewidth=2)
    plt.title("Training Loss Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def get_shapelet_importance(lts_model):
    if lts_model.W.ndim == 3:
        return np.sum(np.abs(lts_model.W), axis=0)   # (R, K)
    return np.abs(lts_model.W)


def plot_top_shapelets(lts_model, filepath, top_n=5):
    importance_matrix = get_shapelet_importance(lts_model)
    flat_importance = importance_matrix.flatten()
    top_idx = np.argsort(flat_importance)[-top_n:][::-1]

    fig, axes = plt.subplots(1, len(top_idx), figsize=(4 * len(top_idx), 4))
    if len(top_idx) == 1:
        axes = [axes]

    for ax, idx in zip(axes, top_idx):
        r, k = np.unravel_index(idx, importance_matrix.shape)
        L_r = (r + 1) * lts_model.L_min_len
        shapelet = lts_model.S[r, k, :L_r]
        shapelet_vis = (shapelet - shapelet.mean()) / (shapelet.std() + 1e-8)

        ax.plot(shapelet_vis, linewidth=3)
        ax.set_title(f"Scale {r+1}\nImp={flat_importance[idx]:.3f}")
        ax.grid(alpha=0.3)

    plt.suptitle("Top Learned Shapelets")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def find_best_match(series, shapelet):
    L = len(shapelet)
    distances = [
        np.mean((series[i:i+L] - shapelet) ** 2)
        for i in range(len(series) - L + 1)
    ]
    return int(np.argmin(distances))


def plot_best_shapelet_alignment(lts_model, X, filepath):
    importance_matrix = get_shapelet_importance(lts_model)
    flat_importance = importance_matrix.flatten()
    best_idx = np.argsort(flat_importance)[-1]

    r, k = np.unravel_index(best_idx, importance_matrix.shape)
    L_r = (r + 1) * lts_model.L_min_len
    best_shapelet = lts_model.S[r, k, :L_r]

    series = X[0]
    pos = find_best_match(series, best_shapelet)

    plt.figure(figsize=(8, 4))
    plt.plot(series, label="Series", linewidth=2)
    plt.plot(
        range(pos, pos + len(best_shapelet)),
        best_shapelet,
        linewidth=3,
        label=f"Best shapelet (scale {r+1})"
    )
    plt.title("Best Shapelet Alignment")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_accuracy_vs_iterations(iter_values, accuracies, filepath):
    plt.figure(figsize=(6, 4))
    plt.plot(iter_values, accuracies, marker="o", linewidth=2)
    plt.xscale("log")
    plt.title("Accuracy vs Training Iterations")
    plt.xlabel("max_iter (log scale)")
    plt.ylabel("Test Accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_accuracy_vs_train_fraction(fractions, accuracies, filepath):
    plt.figure(figsize=(6, 4))
    plt.plot(fractions, accuracies, marker="o", linewidth=2)
    plt.title("Accuracy vs Training Size")
    plt.xlabel("Fraction of Training Data")
    plt.ylabel("Test Accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def run_lts(X_train, y_train, X_test, y_test, cfg):
    model = LearningTimeSeriesShapelets(**cfg)

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, y_pred, acc, elapsed


# =========================================================
#  CHARGEMENT DES DONNÉES
# =========================================================
X_train, y_train, X_test, y_test = load_ucr_dataset(TRAIN_PATH, TEST_PATH)
X_train_scaled, X_test_scaled = normalize_with_sklearn(X_train, X_test)

print("=" * 70)
print(f"Dataset: {DATASET_NAME}")
print(f"Train shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")
print(f"Classes    : {np.unique(y_train)}")
print("=" * 70)

print("\nGlobal signal statistics:")
print(f"Mean   : {X_train.mean():.4f}")
print(f"Std    : {X_train.std():.4f}")
print(f"Min    : {X_train.min():.4f}")
print(f"Max    : {X_train.max():.4f}")

save_class_distribution(y_train, os.path.join(FIGDIR, "class_distribution.png"))
save_series_by_class(X_train_scaled, y_train, os.path.join(FIGDIR, "series_by_class.png"))


# =========================================================
# 4) BASELINES
# =========================================================
results = []

# 1NN
knn = KNeighborsClassifier(n_neighbors=1)
start = time.time()
knn.fit(X_train_scaled, y_train)
pred_knn = knn.predict(X_test_scaled)
time_knn = time.time() - start
acc_knn = accuracy_score(y_test, pred_knn)

results.append({
    "method": "1NN",
    "accuracy": acc_knn,
    "time_sec": time_knn
})

save_confusion_matrix(
    y_test, pred_knn,
    "Confusion Matrix - 1NN",
    os.path.join(FIGDIR, "cm_1nn.png")
)

# SVM
svm = SVC()
start = time.time()
svm.fit(X_train_scaled, y_train)
pred_svm = svm.predict(X_test_scaled)
time_svm = time.time() - start
acc_svm = accuracy_score(y_test, pred_svm)

results.append({
    "method": "SVM (RBF)",
    "accuracy": acc_svm,
    "time_sec": time_svm
})

save_confusion_matrix(
    y_test, pred_svm,
    "Confusion Matrix - SVM",
    os.path.join(FIGDIR, "cm_svm.png")
)


# =========================================================
# LTS (MÉTHODE PRINCIPALE, FIDÈLE AU PAPIER)
# =========================================================
lts, pred_lts, acc_lts, time_lts = run_lts(
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    PAPER_CFG
)

results.append({
    "method": "LTS (paper setting)",
    "accuracy": acc_lts,
    "time_sec": time_lts
})

print(f"\nLTS accuracy: {acc_lts:.4f}")
print(classification_report(y_test, pred_lts))

save_confusion_matrix(
    y_test, pred_lts,
    "Confusion Matrix - LTS",
    os.path.join(FIGDIR, "cm_lts.png")
)

if hasattr(lts, "loss_history") and lts.loss_history is not None:
    plot_loss_curve(
        lts.loss_history,
        os.path.join(FIGDIR, "training_loss.png")
    )

print("S shape:", lts.S.shape)
print("W shape:", lts.W.shape)

plot_top_shapelets(
    lts,
    os.path.join(FIGDIR, "top_shapelets.png"),
    top_n=5
)

plot_best_shapelet_alignment(
    lts,
    X_test_scaled,
    os.path.join(FIGDIR, "best_shapelet_alignment.png")
)


# =========================================================
# ÉTUDE DE CONVERGENCE (PAS DU TUNING)
# =========================================================
iter_values = [50, 100, 200, 500, 1000]
acc_iter = []

for it in iter_values:
    cfg_it = PAPER_CFG.copy()
    cfg_it["max_iter"] = it

    _, _, acc_tmp, _ = run_lts(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        cfg_it
    )
    acc_iter.append(acc_tmp)

plot_accuracy_vs_iterations(
    iter_values,
    acc_iter,
    os.path.join(FIGDIR, "accuracy_vs_iterations.png")
)


# =========================================================
# ÉTUDE DE LA TAILLE DES SHAPELETS
# =========================================================
L_values = [0.1, 0.15, 0.2, 0.25, 0.3]
acc_L = []

for L_val in L_values:
    cfg_L = PAPER_CFG.copy()
    cfg_L["L_min"] = L_val

    _, _, acc_tmp, _ = run_lts(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        cfg_L
    )
    acc_L.append(acc_tmp)

plt.figure(figsize=(6, 4))
plt.plot(L_values, acc_L, marker="o", linewidth=2)
plt.title("Accuracy vs Shapelet Length")
plt.xlabel("L_min (fraction of series length)")
plt.ylabel("Test Accuracy")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "accuracy_vs_shapelet_length.png"), dpi=300)
plt.close()


# =========================================================
# TABLEAU RÉCAPITULATIF
# =========================================================
results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)

print("\n=== Benchmark results ===")
print(results_df)

results_df.to_csv(
    os.path.join(TABDIR, "benchmark_results.csv"),
    index=False
)

with open(os.path.join(LOGDIR, "run_config.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "seed": SEED,
            "dataset_name": DATASET_NAME,
            "train_path": TRAIN_PATH,
            "test_path": TEST_PATH,
            "paper_cfg": PAPER_CFG,
            "iter_values": iter_values,
            "fractions": fractions
        },
        f,
        indent=2
    )

with open(os.path.join(LOGDIR, "classification_report_lts.txt"), "w", encoding="utf-8") as f:
    f.write(classification_report(y_test, pred_lts))

print(f"\nSaved outputs to: {OUTDIR}")
print("Done.")