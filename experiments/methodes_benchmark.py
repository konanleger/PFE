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
from sklearn.linear_model import LogisticRegression

from src.lts import LearningTimeSeriesShapelets
from src.utils import load_ucr_dataset, normalize_with_sklearn


# =========================================================
# 1. CONFIGURATION
# =========================================================
SEED = 42
np.random.seed(SEED)

# Configuration inspirée de l'article pour ItalyPowerDemand
PAPER_CFG = dict(
    K=0.3,
    L_min=0.2,
    R=3,
    lambda_w=0.01,
    learning_rate=0.01,
    max_iter=50,   # mettez 5000 si votre implémentation le supporte en temps
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


# =========================================================
# 2. FONCTIONS UTILITAIRES
# =========================================================
def save_confusion(y_true, y_pred, title, filename):
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


def plot_loss_curve(loss_history, filename):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, linewidth=2)
    plt.title("Training Loss Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_accuracy_vs_train_fraction(fractions, accs, filename):
    plt.figure(figsize=(6, 4))
    plt.plot(fractions, accs, marker="o", linewidth=2)
    plt.title("Accuracy vs Training Size")
    plt.xlabel("Fraction of training data")
    plt.ylabel("Test accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_accuracy_vs_iterations(iter_values, acc_values, filename):
    plt.figure(figsize=(6, 4))
    plt.plot(iter_values, acc_values, marker="o", linewidth=2)
    plt.xscale("log")
    plt.title("Accuracy vs Training Iterations")
    plt.xlabel("max_iter (log scale)")
    plt.ylabel("Test accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def get_shapelet_importance(lts_model):
    if lts_model.W.ndim == 3:
        # somme des poids absolus sur les classes
        importance = np.sum(np.abs(lts_model.W), axis=0)   # (R, K)
    else:
        importance = np.abs(lts_model.W)
    return importance


def plot_top_shapelets(lts_model, filename, top_n=5):
    importance = get_shapelet_importance(lts_model)
    flat = importance.flatten()
    top_idx = np.argsort(flat)[-top_n:][::-1]

    fig, axes = plt.subplots(1, len(top_idx), figsize=(4 * len(top_idx), 4))
    if len(top_idx) == 1:
        axes = [axes]

    for ax, idx in zip(axes, top_idx):
        r, k = np.unravel_index(idx, importance.shape)
        L_r = (r + 1) * lts_model.L_min_len
        sh = lts_model.S[r, k, :L_r]
        sh_vis = (sh - sh.mean()) / (sh.std() + 1e-8)

        ax.plot(sh_vis, linewidth=3)
        ax.set_title(f"Scale {r+1}\nImp={flat[idx]:.3f}")
        ax.grid(alpha=0.3)

    plt.suptitle("Top Learned Shapelets")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def find_best_match(series, shapelet):
    L = len(shapelet)
    distances = []
    for i in range(len(series) - L + 1):
        seg = series[i:i+L]
        distances.append(np.mean((seg - shapelet) ** 2))
    return int(np.argmin(distances))


def plot_best_shapelet_alignment(lts_model, X, filename):
    importance = get_shapelet_importance(lts_model)
    flat = importance.flatten()
    best_idx = np.argsort(flat)[-1]

    r, k = np.unravel_index(best_idx, importance.shape)
    L_r = (r + 1) * lts_model.L_min_len
    best_shapelet = lts_model.S[r, k, :L_r]

    # choisir une série test arbitraire (ici la première)
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
    plt.title("Best Shapelet Alignment Example")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def run_lts(X_train, y_train, X_test, y_test, cfg):
    model = LearningTimeSeriesShapelets(**cfg)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, y_pred, acc, train_time


# =========================================================
# 3. CHARGEMENT DES DONNÉES
# =========================================================
X_train, y_train, X_test, y_test = load_ucr_dataset(
    DATASET["train_path"],
    DATASET["test_path"]
)
X_train_scaled, X_test_scaled = normalize_with_sklearn(X_train, X_test)

print("=" * 70)
print(f"Dataset: {DATASET['name']}")
print(f"Train shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")
print(f"Classes    : {np.unique(y_train)}")
print("=" * 70)

# Sauvegarde EDA
save_class_distribution(
    y_train,
    os.path.join(FIGDIR, "class_distribution.png")
)
save_series_by_class(
    X_train_scaled,
    y_train,
    os.path.join(FIGDIR, "series_by_class.png")
)

# =========================================================
# 4. BASELINES
# =========================================================
results = []

# ---- 1NN brut ----
knn = KNeighborsClassifier(n_neighbors=1)
start = time.time()
knn.fit(X_train_scaled, y_train)
pred_knn = knn.predict(X_test_scaled)
time_knn = time.time() - start
acc_knn = accuracy_score(y_test, pred_knn)

results.append({
    "method": "1NN",
    "accuracy": acc_knn,
    "train_test_time_sec": time_knn
})

save_confusion(
    y_test, pred_knn,
    "Confusion Matrix - 1NN",
    os.path.join(FIGDIR, "cm_1nn.png")
)

# ---- SVM brut ----
svm = SVC()
start = time.time()
svm.fit(X_train_scaled, y_train)
pred_svm = svm.predict(X_test_scaled)
time_svm = time.time() - start
acc_svm = accuracy_score(y_test, pred_svm)

results.append({
    "method": "SVM (RBF)",
    "accuracy": acc_svm,
    "train_test_time_sec": time_svm
})

save_confusion(
    y_test, pred_svm,
    "Confusion Matrix - SVM",
    os.path.join(FIGDIR, "cm_svm.png")
)

# =========================================================
# 5. LTS (CONFIG FIXE PAPIER)
# =========================================================
lts, pred_lts, acc_lts, time_lts = run_lts(
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    PAPER_CFG
)

results.append({
    "method": "LTS (paper setting)",
    "accuracy": acc_lts,
    "train_test_time_sec": time_lts
})

save_confusion(
    y_test, pred_lts,
    "Confusion Matrix - LTS",
    os.path.join(FIGDIR, "cm_lts.png")
)

if hasattr(lts, "loss_history") and lts.loss_history is not None:
    plot_loss_curve(
        lts.loss_history,
        os.path.join(FIGDIR, "loss_curve.png")
    )

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
# 6. LTS TRANSFORM + LOGISTIC REGRESSION
# =========================================================
Xtr_lts = lts.transform(X_train_scaled)
Xte_lts = lts.transform(X_test_scaled)

Xtr_lts = np.nan_to_num(Xtr_lts, nan=np.nanmean(Xtr_lts))
Xte_lts = np.nan_to_num(Xte_lts, nan=np.nanmean(Xte_lts))

logreg = LogisticRegression(max_iter=500)
start = time.time()
logreg.fit(Xtr_lts, y_train)
pred_lr = logreg.predict(Xte_lts)
time_lr = time.time() - start
acc_lr = accuracy_score(y_test, pred_lr)

results.append({
    "method": "LTS-transform + LogReg",
    "accuracy": acc_lr,
    "train_test_time_sec": time_lr
})

save_confusion(
    y_test, pred_lr,
    "Confusion Matrix - LTS-transform + Logistic Regression",
    os.path.join(FIGDIR, "cm_lts_logreg.png")
)

# =========================================================
# 7. ÉTUDE DE CONVERGENCE EN NOMBRE D'ITÉRATIONS
# =========================================================
iter_values = [20, 50]
acc_iter = []

for it in iter_values:
    cfg_it = PAPER_CFG.copy()
    cfg_it["max_iter"] = it

    lts_tmp, pred_tmp, acc_tmp, _ = run_lts(
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
# 8. ÉTUDE DE LA TAILLE D'APPRENTISSAGE
# =========================================================
fractions = [0.1, 0.3]
acc_fraction = []

for frac in fractions:
    n = max(2, int(frac * len(X_train_scaled)))

    lts_tmp, pred_tmp, acc_tmp, _ = run_lts(
        X_train_scaled[:n], y_train[:n],
        X_test_scaled, y_test,
        PAPER_CFG
    )
    acc_fraction.append(acc_tmp)

plot_accuracy_vs_train_fraction(
    fractions,
    acc_fraction,
    os.path.join(FIGDIR, "accuracy_vs_train_fraction.png")
)

# =========================================================
# 9. SAUVEGARDE DES RÉSULTATS
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
            "dataset": DATASET,
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