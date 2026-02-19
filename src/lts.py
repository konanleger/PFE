import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from scipy.special import expit  # sigmoid function
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class LearningTimeSeriesShapelets:
    """
    Implémentation de la méthode "Learning Time-Series Shapelets" (LTS)
    basée sur l'article de Grabocka et al.
    """
    
    def __init__(self, K=0.3, L_min=0.1, R=3, lambda_w=0.01, 
                 learning_rate=0.01, max_iter=5000, alpha=-100):
        """
        Initialisation des paramètres
        
        Parameters:
        -----------
        K : float ou int
            Nombre de shapelets (fraction de Q si float < 1)
        L_min : float
            Longueur minimale des shapelets (fraction de Q)
        R : int
            Nombre d'échelles de longueurs
        lambda_w : float
            Paramètre de régularisation
        learning_rate : float
            Taux d'apprentissage
        max_iter : int
            Nombre maximal d'itérations
        alpha : float
            Paramètre de précision pour soft-min (négatif)
        """
        self.K = K
        self.L_min = L_min
        self.R = R
        self.lambda_w = lambda_w
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        
        # Paramètres appris
        self.S = None  # Shapelets
        self.W = None  # Poids de classification
        self.W0 = None  # Biais
        self.classes_ = None
        self.L_min_len = None  # longueur de base des shapelets
        
    def _soft_minimum(self, D):
        """
        Fonction soft-minimum différentiable (Eq. 6, 19)
        
        Parameters:
        -----------
        D : numpy array
            Distances entre shapelet et segments
            
        Returns:
        --------
        M_tilde : float
            Approximation soft-min
        """
        # Éviter les overflow numériques
        D_max = np.max(D)
        exp_terms = np.exp(self.alpha * (D - D_max))
        
        numerator = np.sum(D * exp_terms)
        denominator = np.sum(exp_terms)
        
        if denominator == 0:
            return np.min(D)
        
        return numerator / denominator
    
    def _compute_distances(self, series, shapelets, L):
        """
        Calcule les distances entre une série et des shapelets
        
        Parameters:
        -----------
        series : numpy array (Q,)
            Une série temporelle
        shapelets : numpy array (K, L) ou (R, K, L)
            Shapelets
        L : int
            Longueur des shapelets
            
        Returns:
        --------
        distances : numpy array
            Distances minimales
        """
        Q = len(series)
        J = Q - L + 1
        
        if len(shapelets.shape) == 2:  # Cas simple
            K = shapelets.shape[0]
            distances = np.zeros(K)
            
            for k in range(K):
                # Calculer les distances pour tous les segments
                segment_dists = np.zeros(J)
                for j in range(J):
                    segment = series[j:j+L]
                    segment_dists[j] = np.mean((segment - shapelets[k]) ** 2)
                
                # Appliquer soft-minimum
                distances[k] = self._soft_minimum(segment_dists)
            
            return distances
        
        else:  # Cas multi-échelles (R, K, L)
            R, K, _ = shapelets.shape
            distances = np.zeros((R, K))
            
            for r in range(R):
                L_r = shapelets.shape[2]
                J_r = Q - L_r + 1
                
                for k in range(K):
                    segment_dists = np.zeros(J_r)
                    for j in range(J_r):
                        segment = series[j:j+L_r]
                        segment_dists[j] = np.mean((segment - shapelets[r, k]) ** 2)
                    
                    distances[r, k] = self._soft_minimum(segment_dists)
            
            return distances
    
    def _initialize_shapelets(self, X, y):
        """
        Initialisation des shapelets avec K-Means (Section 3.9)
        
        Parameters:
        -----------
        X : numpy array (I, Q)
            Données d'entraînement
        y : numpy array (I,)
            Labels
        """
        I, Q = X.shape
        
        # Déterminer K si c'est une fraction
        if isinstance(self.K, float) and self.K < 1:
            K = int(self.K * Q)
        else:
            K = int(self.K)
        
        # Déterminer L_min si c'est une fraction
        if isinstance(self.L_min, float) and self.L_min < 1:
            L_min = int(self.L_min * Q)
        else:
            L_min = int(self.L_min)
        
        # Garder L_min entre des limites raisonnables
        L_min = max(3, min(L_min, Q // 3))
        self.L_min_len = L_min
        
        print(f"Initialisation: K={K}, L_min={L_min}, R={self.R}")
        
        # Extraire tous les segments de longueur L_min
        all_segments = []
        for i in range(I):
            J_i = Q - L_min + 1
            for j in range(J_i):
                all_segments.append(X[i, j:j+L_min])
        
        all_segments = np.array(all_segments)
        
        # Appliquer K-Means pour initialiser les shapelets
        if len(all_segments) > K:
            kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
            kmeans.fit(all_segments)
            
            # Initial shapelets pour l'échelle de base
            S_base = kmeans.cluster_centers_
            
            # Allouer avec longueur maximale
            L_max = L_min * self.R
            self.S = np.zeros((self.R, K, L_max))
            
            # Placer les shapelets de base
            self.S[0, :, :L_min] = S_base[:, :L_min]
            
            # Initialiser les autres échelles par interpolation
            for r in range(1, self.R):
                L_r = (r + 1) * L_min
                if L_r <= Q and L_r <= L_max:
                    for k in range(K):
                        self.S[r, k, :L_r] = np.interp(
                            np.linspace(0, 1, L_r),
                            np.linspace(0, 1, L_min),
                            S_base[k]
                        )
                else:
                    for k in range(K):
                        take_len = min(L_r, L_max, S_base.shape[1])
                        self.S[r, k, :take_len] = S_base[k, :take_len]
        else:
            # Fallback si pas assez de segments
            L_max = L_min * self.R
            self.S = np.random.randn(self.R, K, L_max) * 0.1
        
        # Initialiser les poids
        self.classes_ = np.unique(y)
        C = len(self.classes_)
        
        self.W = np.random.randn(C, self.R, K) * 0.01
        self.W0 = np.random.randn(C) * 0.01
        
        return K, L_min
    
    def _compute_gradients(self, X_i, y_i_b, c):
        """
        Calcul des gradients pour une instance (Eq. 22-26)
        
        Parameters:
        -----------
        X_i : numpy array (Q,)
            Une série temporelle
        y_i_b : int
            Label binaire (0 ou 1) pour la classe c
        c : int
            Indice de la classe
            
        Returns:
        --------
        dS : numpy array
            Gradient des shapelets
        dW : numpy array
            Gradient des poids
        dW0 : float
            Gradient du biais
        """
        if X_i.ndim == 1:
            I, Q = 1, X_i.shape[0]
            X_i = X_i.reshape(1, -1)
        else:
            I, Q = X_i.shape
        
        C, R, K = self.W.shape
        L_min = self.L_min_len
        
        # Calculer les distances M
        M = np.zeros((R, K))
        D_all = []  # Stocker les distances brutes pour le gradient
        
        for r in range(R):
            L_r = (r + 1) * L_min
            if L_r > Q:
                continue
                
            J_r = Q - L_r + 1
            D_r = np.zeros((K, J_r))
            
            for k in range(K):
                for j in range(J_r):
                    segment = X_i[0, j:j+L_r]
                    shapelet_seg = self.S[r, k, :L_r]
                    D_r[k, j] = np.mean((segment - shapelet_seg) ** 2)
                
                # Soft minimum
                M[r, k] = self._soft_minimum(D_r[k])
            
            D_all.append(D_r)
        
        # Prédiction (Eq. 17)
        y_pred = self.W0[c]
        for r in range(R):
            for k in range(K):
                y_pred += M[r, k] * self.W[c, r, k]
        
        # Sigmoid
        sigmoid_val = expit(y_pred)
        
        # Gradient de base (Eq. 9, 25, 26)
        error = y_i_b - sigmoid_val
        
        # Gradient pour W
        dW = np.zeros((R, K))
        for r in range(R):
            for k in range(K):
                dW[r, k] = -error * M[r, k] + (self.lambda_w * self.W[c, r, k]) / (I * C)
        
        # Gradient pour W0
        dW0 = -error
        
        # Gradient pour S (simplifié)
        dS = np.zeros_like(self.S)
        
        for r in range(R):
            L_r = (r + 1) * L_min
            if L_r > Q or r >= len(D_all):
                continue
                
            J_r = Q - L_r + 1
            D_r = D_all[r]
            
            for k in range(K):
                # Calculer les poids exponentiels pour soft-min
                D_k = D_r[k]
                exp_terms = np.exp(self.alpha * D_k)
                sum_exp = np.sum(exp_terms)
                
                if sum_exp == 0:
                    continue
                
                # Gradient du soft-min par rapport à D (Eq. 23 simplifié)
                soft_min_val = M[r, k]
                weights = exp_terms / sum_exp
                
                for j in range(J_r):
                    # Terme principal
                    weight_j = weights[j]
                    
                    # Gradient de D par rapport à S (Eq. 24)
                    segment = X_i[0, j:j+L_r]
                    shapelet_seg = self.S[r, k, :L_r]
                    dD_dS_seg = 2 * (shapelet_seg - segment) / L_r
                    
                    # Contribution totale (ranger dans les premiers L_r éléments)
                    dS[r, k, :L_r] += -error * self.W[c, r, k] * weight_j * dD_dS_seg
        
        return dS, dW, dW0
    
    def fit(self, X, y):
        """
        Entraînement du modèle
        
        Parameters:
        -----------
        X : numpy array (I, Q)
            Données d'entraînement
        y : numpy array (I,)
            Labels
        """
        I, Q = X.shape
        
        # Initialisation
        K, L_min = self._initialize_shapelets(X, y)
        
        # One-vs-all encoding (Eq. 15)
        self.classes_ = np.unique(y)
        C = len(self.classes_)
        
        y_binary = np.zeros((I, C))
        for i, label in enumerate(y):
            class_idx = np.where(self.classes_ == label)[0][0]
            y_binary[i, class_idx] = 1
        
        print(f"Début de l'entraînement: I={I}, Q={Q}, C={C}")
        print(f"Taille des shapelets: {self.S.shape}")
        
        # Entraînement par descente de gradient stochastique
        for iteration in tqdm(range(self.max_iter), desc="Entraînement"):
            # Parcourir toutes les instances
            for i in range(I):
                X_i = X[i]
                
                # Mettre à jour pour chaque classe (one-vs-all)
                for c in range(C):
                    y_i_b = y_binary[i, c]
                    
                    # Calculer les gradients
                    dS, dW, dW0 = self._compute_gradients(X_i, y_i_b, c)
                    
                    # Mettre à jour les paramètres
                    self.S -= self.learning_rate * dS
                    self.W[c] -= self.learning_rate * dW
                    self.W0[c] -= self.learning_rate * dW0
            
            # Réduction du learning rate
            if iteration % 1000 == 0 and iteration > 0:
                self.learning_rate *= 0.9
        
        print("Entraînement terminé")
        return self
    
    def predict_proba(self, X):
        """
        Prédire les probabilités pour chaque classe
        
        Parameters:
        -----------
        X : numpy array (I_test, Q)
            Données de test
            
        Returns:
        --------
        probas : numpy array (I_test, C)
            Probabilités pour chaque classe
        """
        I_test = X.shape[0]
        C = len(self.classes_)
        
        probas = np.zeros((I_test, C))
        
        for i in range(I_test):
            X_i = X[i]
            
            for c in range(C):
                # Calculer les distances
                M = np.zeros((self.R, self.S.shape[1]))
                
                for r in range(self.R):
                    L_r = (r + 1) * self.L_min_len
                    if L_r > len(X_i):
                        continue
                    
                    for k in range(self.S.shape[1]):
                        # Distance minimale
                        J_r = len(X_i) - L_r + 1
                        distances = np.zeros(J_r)
                        
                        for j in range(J_r):
                            segment = X_i[j:j+L_r]
                            shapelet_seg = self.S[r, k, :L_r]
                            distances[j] = np.mean((segment - shapelet_seg) ** 2)
                        
                        M[r, k] = self._soft_minimum(distances)
                
                # Prédiction linéaire
                y_pred = self.W0[c]
                for r in range(self.R):
                    for k in range(self.S.shape[1]):
                        y_pred += M[r, k] * self.W[c, r, k]
                
                # Sigmoid pour probabilité
                probas[i, c] = expit(y_pred)
        
        # Normaliser les probabilités
        probas_sum = probas.sum(axis=1, keepdims=True)
        probas_sum[probas_sum == 0] = 1  # Éviter division par zéro
        probas = probas / probas_sum
        
        return probas
    
    def predict(self, X):
        """
        Prédire les labels
        
        Parameters:
        -----------
        X : numpy array (I_test, Q)
            Données de test
            
        Returns:
        --------
        predictions : numpy array (I_test,)
            Labels prédits
        """
        probas = self.predict_proba(X)
        predictions_idx = np.argmax(probas, axis=1)
        predictions = self.classes_[predictions_idx]
        
        return predictions
    
    def transform(self, X):
        """
        Transformer les séries en distances aux shapelets
        
        Parameters:
        -----------
        X : numpy array (I, Q)
            Séries temporelles
            
        Returns:
        --------
        M_transformed : numpy array (I, R*K)
            Représentation transformée
        """
        I = X.shape[0]
        R, K, _ = self.S.shape
        
        M_transformed = np.zeros((I, R * K))
        
        for i in range(I):
            X_i = X[i]
            distances_flat = []
            
            for r in range(R):
                L_r = (r + 1) * self.L_min_len
                if L_r > len(X_i):
                    distances_flat.extend([np.nan] * K)
                    continue
                
                for k in range(K):
                    J_r = len(X_i) - L_r + 1
                    segment_dists = np.zeros(J_r)
                    
                    for j in range(J_r):
                        segment = X_i[j:j+L_r]
                        shapelet_seg = self.S[r, k, :L_r]
                        segment_dists[j] = np.mean((segment - shapelet_seg) ** 2)
                    
                    M_transformed[i, r*K + k] = self._soft_minimum(segment_dists)
        
        return M_transformed