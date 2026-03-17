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

############ TEST DE BON FONCTIONNEMENT DE LTS SUR LES DATASETS  ####################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from scipy.special import expit

from src.lts import LearningTimeSeriesShapelets
from src.utils import load_ucr_dataset, normalize_with_sklearn

# Charger et normaliser les données
X_train, y_train, X_test, y_test = load_ucr_dataset(
    'data/ItalyPowerDemand_TRAIN',
    'data/ItalyPowerDemand_TEST'
)
X_train_scaled, X_test_scaled = normalize_with_sklearn(X_train, X_test)

print("=" * 70)
print("TESTS DE BON FONCTIONNEMENT DE LA CLASSE LTS")
print("=" * 70)

# --------------------------------------------------
# 1) Initialisation de l'objet
# --------------------------------------------------
lts = LearningTimeSeriesShapelets(
    K=0.3,
    L_min=0.2,
    R=3,
    lambda_w=0.01,
    learning_rate=0.01,
    max_iter=5000,   # petit nombre juste pour test fonctionnel
    alpha=-100
)

print("\n[OK] Objet LTS créé")
print("Paramètres :", {
    "K": lts.K,
    "L_min": lts.L_min,
    "R": lts.R,
    "lambda_w": lts.lambda_w,
    "learning_rate": lts.learning_rate,
    "max_iter": lts.max_iter,
    "alpha": lts.alpha
})

# --------------------------------------------------
# 2) Test _soft_minimum
# --------------------------------------------------
D_test = np.array([0.4, 0.2, 0.8, 0.1, 0.5])
softmin_val = lts._soft_minimum(D_test)

print("\n--- Test _soft_minimum ---")
print("Distances test :", D_test)
print("min exact      :", np.min(D_test))
print("soft-min       :", softmin_val)

assert np.isfinite(softmin_val), "_soft_minimum retourne une valeur non finie"
print("[OK] _soft_minimum fonctionne")

# --------------------------------------------------
# 3) Test _initialize_shapelets
# --------------------------------------------------
print("\n--- Test _initialize_shapelets ---")
K_real, L_min_real = lts._initialize_shapelets(X_train_scaled, y_train)

print("K réel         :", K_real)
print("L_min réel     :", L_min_real)
print("Shape S        :", lts.S.shape)
print("Weights W      :", lts.W.shape)
print("Bias W0        :", lts.W0.shape)
print("Classes        :", lts.classes_)

assert lts.S is not None, "S non initialisé"
assert lts.W is not None, "W non initialisé"
assert lts.W0 is not None, "W0 non initialisé"
assert lts.L_min_len is not None, "L_min_len non initialisé"
print("[OK] _initialize_shapelets fonctionne")

# --------------------------------------------------
# 4) Test _compute_distances
# --------------------------------------------------
print("\n--- Test _compute_distances ---")
series0 = X_train_scaled[0]
L0 = lts.L_min_len
shapelets_scale0 = lts.S[0, :, :L0]   # shape (K, L0)

distances = lts._compute_distances(series0, shapelets_scale0, L0)

print("Shape des distances :", distances.shape)
print("Distances (extrait) :", distances[:min(5, len(distances))])

assert distances.ndim == 1, "_compute_distances devrait retourner un vecteur ici"
assert np.all(np.isfinite(distances)), "Distances non finies"
print("[OK] _compute_distances fonctionne")

# --------------------------------------------------
# 5) Test _compute_gradients
# --------------------------------------------------
print("\n--- Test _compute_gradients ---")
# Construire y_binary comme dans fit
classes_ = np.unique(y_train)
C = len(classes_)
y_binary = np.zeros((len(y_train), C))
for i, label in enumerate(y_train):
    class_idx = np.where(classes_ == label)[0][0]
    y_binary[i, class_idx] = 1

X_i = X_train_scaled[0]
c = 0
y_i_b = y_binary[0, c]

out = lts._compute_gradients(X_i, y_i_b, c)

print("Nombre d'éléments retournés :", len(out))
assert len(out) == 4, "_compute_gradients devrait retourner dS, dW, dW0, y_pred"

dS, dW, dW0, y_pred = out

print("dS shape :", dS.shape)
print("dW shape :", dW.shape)
print("dW0      :", dW0)
print("y_pred   :", y_pred)

assert dS.shape == lts.S.shape, "Shape de dS incorrecte"
assert dW.shape == lts.W[c].shape, "Shape de dW incorrecte"
assert np.isfinite(dW0), "dW0 non fini"
assert np.isfinite(y_pred), "y_pred non fini"
print("[OK] _compute_gradients fonctionne")

# --------------------------------------------------
# 6) Test fit
# --------------------------------------------------
print("\n--- Test fit ---")
lts_fit = LearningTimeSeriesShapelets(
    K=0.3,
    L_min=0.2,
    R=3,
    lambda_w=0.01,
    learning_rate=0.01,
    max_iter=500,   # test rapide
    alpha=-100
)
# Initialiser sans entraîner
lts_fit._initialize_shapelets(X_train_scaled, y_train)
# Sauvegarder shapelets initiaux
S_before = lts_fit.S.copy()

# Entraîner
lts_fit.fit(X_train_scaled, y_train)
# Sauvegarder shapelets après entraînement
S_after = lts_fit.S.copy()

assert lts_fit.S is not None, "fit n'a pas appris S"
assert lts_fit.W is not None, "fit n'a pas appris W"
assert lts_fit.W0 is not None, "fit n'a pas appris W0"
assert hasattr(lts_fit, "loss_history"), "loss_history absent"
assert len(lts_fit.loss_history) == lts_fit.max_iter, "loss_history incomplet"

print("loss_history (5 premières valeurs) :", lts_fit.loss_history[:5])
print("[OK] fit fonctionne")

# --------------------------------------------------
# 7) Test predict_proba
# --------------------------------------------------
print("\n--- Test predict_proba ---")
probas = lts_fit.predict_proba(X_test_scaled[:10])

print("Shape probas :", probas.shape)
print("Somme par ligne :", probas.sum(axis=1))

assert probas.shape == (10, len(lts_fit.classes_)), "Shape de predict_proba incorrecte"
assert np.all(np.isfinite(probas)), "predict_proba contient des NaN/Inf"
assert np.allclose(probas.sum(axis=1), 1, atol=1e-5), "Les probabilités ne somment pas à 1"
print("[OK] predict_proba fonctionne")

# --------------------------------------------------
# 8) Test predict
# --------------------------------------------------
print("\n--- Test predict ---")
y_pred_test = lts_fit.predict(X_test_scaled[:20])

print("Prédictions :", y_pred_test)
print("Labels possibles :", lts_fit.classes_)

assert y_pred_test.shape == (20,), "Shape de predict incorrecte"
assert np.all(np.isin(y_pred_test, lts_fit.classes_)), "predict retourne des labels invalides"
print("[OK] predict fonctionne")

# --------------------------------------------------
# 9) Test transform
# --------------------------------------------------
print("\n--- Test transform ---")
X_transformed = lts_fit.transform(X_test_scaled[:10])

print("Shape transform :", X_transformed.shape)
print("Extrait transform :")
print(X_transformed[:2])

R, K_real, _ = lts_fit.S.shape
assert X_transformed.shape == (10, R * K_real), "Shape de transform incorrecte"
assert np.all(np.isfinite(X_transformed)), "transform contient des NaN/Inf"
print("[OK] transform fonctionne")


print("\n--- Test update des shapelets ---")
# Mesure de différence
shapelet_diff = np.linalg.norm(S_after - S_before)
max_abs_diff = np.max(np.abs(S_after - S_before))

print(f"Norme de différence ||S_after - S_before|| = {shapelet_diff:.6f}")
print(f"Différence absolue max                 = {max_abs_diff:.6f}")

# Vérification
assert shapelet_diff > 0, "Les shapelets n'ont pas été mis à jour"
assert max_abs_diff > 0, "Aucune valeur de shapelet n'a changé"

print("[OK] Les shapelets sont bien mis à jour pendant fit()")


print("\n--- Visualisation avant/après d'un shapelet ---")

r_test, k_test = 0, 0
L_r = (r_test + 1) * lts_fit.L_min_len

plt.figure(figsize=(8, 4))
plt.plot(S_before[r_test, k_test, :L_r], label="Avant entraînement", linewidth=2)
plt.plot(S_after[r_test, k_test, :L_r], label="Après entraînement", linewidth=2)
plt.title("Évolution d'un shapelet pendant l'entraînement")
plt.xlabel("Temps")
plt.ylabel("Valeur")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

changed_mask = np.abs(S_after - S_before) > 1e-12
n_changed = np.sum(np.any(changed_mask, axis=2))  # nombre de shapelets modifiés

print(f"Nombre de shapelets modifiés : {n_changed} / {S_before.shape[0] * S_before.shape[1]}")
assert n_changed > 0, "Aucun shapelet n'a été modifié"
print("[OK] Plusieurs shapelets ont été modifiés pendant l'entraînement")

# --------------------------------------------------
# 10) Petit test global de performance
# --------------------------------------------------
print("\n--- Test global pipeline ---")
y_pred_full = lts_fit.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred_full)

print(f"Accuracy test (max_iter={lts_fit.max_iter}) : {acc:.4f}")
print(classification_report(y_test, y_pred_full))

cm = confusion_matrix(y_test, y_pred_full)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title("Confusion Matrix - Test fonctionnel LTS")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("TOUS LES TESTS SONT TERMINÉS")
print("=" * 70)