import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from src.lts import LearningTimeSeriesShapelets
from src.utils import load_ucr_dataset, normalize_with_sklearn

# Charger et normaliser les données
X_train, y_train, X_test, y_test = load_ucr_dataset(
    'data/ItalyPowerDemand_TRAIN',
    'data/ItalyPowerDemand_TEST'
)
X_train_scaled, X_test_scaled = normalize_with_sklearn(X_train, X_test)

#  Initialiser et entraîner le modèle
lts = LearningTimeSeriesShapelets(K=0.3, L_min=0.2, R=3, max_iter=20)
lts.fit(X_train_scaled, y_train)

# Évaluer le modèle
y_pred = lts.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy test: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Visualiser quelques shapelets
# fig, axes = plt.subplots(lts.R, min(3, lts.S.shape[1]), figsize=(12, 4))
# if lts.R == 1:
#     axes = axes.reshape(1, -1)
# for r in range(lts.R):
#     for k in range(min(3, lts.S.shape[1])):
#         axes[r, k].plot(lts.S[r, k])
#         axes[r, k].set_title(f'Échelle {r+1}, Shapelet {k+1}')
# plt.tight_layout()
# plt.savefig('results/shapelets_appris.png')
# plt.show()


# Répartition des classes
class_counts = pd.Series(y_train).value_counts().sort_index()
print("Class distribution:")
print(class_counts)

# Plot distribution
plt.figure(figsize=(6,4))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Class Distribution (Training)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/class_distribution.png", dpi=300)
plt.show()

print("\nGlobal signal statistics:")
print(f"Mean   : {X_train.mean():.4f}")
print(f"Std    : {X_train.std():.4f}")
print(f"Min    : {X_train.min():.4f}")
print(f"Max    : {X_train.max():.4f}")


plt.figure(figsize=(10,5))

for c in np.unique(y_train):
    idx = np.where(y_train == c)[0][:10]  # 10 séries par classe
    for i in idx:
        plt.plot(X_train_scaled[i], alpha=0.3)

    mean_series = X_train_scaled[y_train == c].mean(axis=0)
    plt.plot(mean_series, linewidth=3, label=f"Mean class {c}")

plt.title("Time Series by Class (Train)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/series_by_class.png", dpi=300)
plt.show()



cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.show()


# Dimensions
print("S shape:", lts.S.shape)
print("W shape:", lts.W.shape)

# Importance = somme des poids absolus sur les classes
if lts.W.ndim == 3:
    # cas multi-classe : (C, R, K)
    importance_matrix = np.sum(np.abs(lts.W), axis=0)
else:
    # cas binaire simple : (R, K)
    importance_matrix = np.abs(lts.W)

# Flatten proprement
flat_importance = importance_matrix.flatten()

# Top 5
top_idx = np.argsort(flat_importance)[-5:][::-1]
print("Top shapelets indices:", top_idx)

fig, axes = plt.subplots(1, len(top_idx), 
                         figsize=(4*len(top_idx), 4))

if len(top_idx) == 1:
    axes = [axes]

for ax, idx in zip(axes, top_idx):

    r, k = np.unravel_index(idx, importance_matrix.shape)

    shapelet = lts.S[r, k]
    shapelet_norm = (shapelet - shapelet.mean()) / (shapelet.std() + 1e-8)

    ax.plot(shapelet_norm, linewidth=3)
    ax.set_title(f"Scale {r+1}, Importance={flat_importance[idx]:.3f}")
    ax.grid(alpha=0.3)

plt.suptitle("Top Learned Shapelets")
plt.tight_layout()
plt.savefig("results/top_shapelets.png", dpi=300)
plt.show()


def find_best_match(series, shapelet):
    L = len(shapelet)
    distances = [
        np.mean((series[i:i+L] - shapelet)**2)
        for i in range(len(series) - L + 1)
    ]
    return np.argmin(distances)

# Visualiser sur une série test
series = X_test_scaled[0]
best_shapelet = lts.S[0, 0]

pos = find_best_match(series, best_shapelet)

plt.figure(figsize=(8,4))
plt.plot(series, label="Series")
plt.plot(range(pos, pos+len(best_shapelet)),
         best_shapelet,
         linewidth=3,
         label="Best match")
plt.legend()
plt.title("Shapelet Alignment Example")
plt.grid(alpha=0.3)
plt.savefig("results/shapelet_alignment.png", dpi=300)
plt.show()




############### amelioration de la visualisation des shapelets appris ###############

# Ajouter une analyse de convergence
plt.figure(figsize=(6,4))
plt.plot(lts.loss_history)
plt.title("Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.savefig("results/training_loss.png", dpi=300)
plt.show()

### Étude de sensibilité aux hyperparamètres
Ks = [0.1, 0.2, 0.3, 0.4]
results = []

for k_val in Ks:
    lts = LearningTimeSeriesShapelets(K=k_val, L_min=0.2, R=3, max_iter=30)
    lts.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, lts.predict(X_test_scaled))
    results.append(acc)

plt.plot(Ks, results, marker='o')
plt.title("Accuracy vs Number of Shapelets")
plt.xlabel("K (fraction of Q)")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig("results/sensitivity_K.png", dpi=300)
plt.show()

# Étude de la taille d’apprentissage

fractions = [0.3, 0.5, 0.7, 1.0]
acc_train_size = []

for frac in fractions:
    n = int(frac * len(X_train_scaled))
    lts = LearningTimeSeriesShapelets(K=0.3, L_min=0.2, R=3, max_iter=30)
    lts.fit(X_train_scaled[:n], y_train[:n])
    acc = accuracy_score(y_test, lts.predict(X_test_scaled))
    acc_train_size.append(acc)

plt.plot(fractions, acc_train_size, marker='o')
plt.title("Accuracy vs Training Size")
plt.xlabel("Fraction of Training Data")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig("results/sensitivity_train_size.png", dpi=300)
plt.show()

# Comparaison avec baselines

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 1-NN brut
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_scaled, y_train)
acc_knn = accuracy_score(y_test, knn.predict(X_test_scaled))

# SVM brut
svm = SVC()
svm.fit(X_train_scaled, y_train)
acc_svm = accuracy_score(y_test, svm.predict(X_test_scaled))

print("Baseline 1NN:", acc_knn)
print("Baseline SVM:", acc_svm)
print("LTS:", accuracy)

# Améliorer la visualisation des shapelets appris
mean_class1 = X_train_scaled[y_train == 1].mean(axis=0)
mean_class2 = X_train_scaled[y_train == 2].mean(axis=0)

plt.plot(mean_class1, label="Mean class 1")
plt.plot(mean_class2, label="Mean class 2")
plt.legend()
plt.title("Class Mean Comparison")
plt.savefig("results/class_means.png", dpi=300)
plt.show()

# Ajouter un tableau résumé des résultats
results_df = pd.DataFrame({
    "Method": ["1NN", "SVM", "LTS"],
    "Accuracy": [acc_knn, acc_svm, accuracy]
})

print(results_df)
results_df.to_csv("results/summary_results.csv", index=False)