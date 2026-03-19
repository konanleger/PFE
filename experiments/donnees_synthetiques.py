import numpy as np
import matplotlib.pyplot as plt

from src.lts import LearningTimeSeriesShapelets
from src.utils import normalize_with_sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

SEED = 42
np.random.seed(SEED)
# -----------------------------
# Génération du dataset
# -----------------------------
import numpy as np

def generate_easy_synthetic(n_samples=200, length=40, noise_std=0.05):
    X = []
    y = []

    motif_pos = 17
    motif_len = 6

    for i in range(n_samples):
        signal = np.random.normal(0, noise_std, length)

        if i < n_samples // 2:
            signal[motif_pos:motif_pos+motif_len] += np.array([0, 1, 2, 2, 1, 0])
            y.append(0)
        else:
            signal[motif_pos:motif_pos+motif_len] += np.array([0, -1, -2, -2, -1, 0])
            y.append(1)

        X.append(signal)

    return np.array(X), np.array(y)


X_train, y_train = generate_easy_synthetic(200)
X_test, y_test = generate_easy_synthetic(200)

X_train_scaled, X_test_scaled = normalize_with_sklearn(X_train, X_test)

lts = LearningTimeSeriesShapelets(
    K=2,
    L_min=6,
    R=1,
    lambda_w=0.01,
    learning_rate=0.01,
    max_iter=200,
    alpha=-100
)

lts.fit(X_train_scaled, y_train)
y_pred = lts.predict(X_test_scaled)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))