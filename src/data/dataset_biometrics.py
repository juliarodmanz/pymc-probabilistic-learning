import numpy as np

def generate_biometric_data(n_samples=150):
    """
    Genera el dataset sintético para el problema de autenticación biométrica (Clasificación).
    Retorna los datos de entrenamiento (In-Distribution) y los datos del atacante (OOD).
    """
    np.random.seed(2003)

    # Clase 0: Usuario Autorizado
    mu_0 = [1.5, 2.0]
    cov_0 = [[0.3, 0.1], [0.1, 0.4]]
    X_0 = np.random.multivariate_normal(mu_0, cov_0, n_samples)
    y_0 = np.zeros(n_samples)

    # Clase 1: Familiar A 
    mu_1 = [-1.5, 1.5]
    cov_1 = [[0.4, -0.1], [-0.1, 0.3]]
    X_1 = np.random.multivariate_normal(mu_1, cov_1, n_samples)
    y_1 = np.ones(n_samples)

    # Clase 2: Familiar B 
    mu_2 = [0.0, -1.5]
    cov_2 = [[0.5, 0.2], [0.2, 0.3]]
    X_2 = np.random.multivariate_normal(mu_2, cov_2, n_samples)
    y_2 = np.full(n_samples, 2)

    X_train = np.vstack((X_0, X_1, X_2))
    y_train = np.concatenate((y_0, y_1, y_2)).astype(int)

    # Atacante (OOD)
    mu_ood = [5.0, -2.5]
    cov_ood = [[0.6, 0.0], [0.0, 0.6]]
    X_ood = np.random.multivariate_normal(mu_ood, cov_ood, 30)
    y_ood = np.full(30, -1) # Etiqueta dummy para el atacante

    return X_train, y_train, X_ood, y_ood