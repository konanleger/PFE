import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_ucr_dataset(train_path, test_path=None):
    """
    Charger un dataset UCR/UEA
    
    Parameters:
    -----------
    train_path : str
        Chemin vers le fichier d'entraînement
    test_path : str, optional
        Chemin vers le fichier de test
        
    Returns:
    --------
    X_train, y_train, X_test, y_test (si test_path fourni)
    """
    # Charger les données d'entraînement
    train_data = pd.read_csv(train_path, header=None, sep=',')
    y_train = train_data.iloc[:, 0].values
    X_train = train_data.iloc[:, 1:].values
    
    print(f"Train: {X_train.shape} séries, {len(np.unique(y_train))} classes")
    
    if test_path:
        # Charger les données de test
        test_data = pd.read_csv(test_path, header=None, sep=',')
        y_test = test_data.iloc[:, 0].values
        X_test = test_data.iloc[:, 1:].values
        
        print(f"Test: {X_test.shape} séries")
        
        return X_train, y_train, X_test, y_test
    
    return X_train, y_train

def normalize_with_sklearn(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    return X_train_scaled