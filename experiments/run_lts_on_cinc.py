import numpy as np
import matplotlib.pyplot as plt
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
lts = LearningTimeSeriesShapelets(K=0.3, L_min=0.2, R=3, max_iter=5000)
lts.fit(X_train_scaled, y_train)

# Évaluer le modèle
y_pred = lts.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy test: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Visualiser quelques shapelets
fig, axes = plt.subplots(lts.R, min(3, lts.S.shape[1]), figsize=(12, 4))
if lts.R == 1:
    axes = axes.reshape(1, -1)
for r in range(lts.R):
    for k in range(min(3, lts.S.shape[1])):
        axes[r, k].plot(lts.S[r, k])
        axes[r, k].set_title(f'Échelle {r+1}, Shapelet {k+1}')
plt.tight_layout()
plt.savefig('results/shapelets_appris.png')
plt.show()
