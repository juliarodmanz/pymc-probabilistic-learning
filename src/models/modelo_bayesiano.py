import pymc as pm
import pytensor.tensor as pt
import numpy as np

def build_and_train_bnn(X_train, y_train, n_hidden=50, inference_method='advi'):
    """
    Construcción y entrenamiento de una Red Neuronal Bayesiana usando PyMC.
    Arquitectura corregida con escalado de priors para evitar la saturación de gradientes.
    """
    X_train_matrix = X_train[:, None]
    
    with pm.Model() as bnn_model:
        # Datos compartidos
        X_shared = pm.Data('X', X_train_matrix)
        y_shared = pm.Data('y', y_train)

        # --- CAPA OCULTA 1 (Entrada: 1 -> Oculta: n_hidden) ---
        w1 = pm.Normal('w1', mu=0, sigma=1.0, shape=(1, n_hidden))
        b1 = pm.Normal('b1', mu=0, sigma=1.0, shape=(n_hidden,))

        # --- CAPA OCULTA 2 (Entrada: n_hidden -> Oculta: n_hidden) ---
        # He/Xavier probabilístico: escalamos la incertidumbre inicial según las neuronas de entrada
        sigma_w2 = np.sqrt(2.0 / n_hidden) 
        w2 = pm.Normal('w2', mu=0, sigma=sigma_w2, shape=(n_hidden, n_hidden))
        b2 = pm.Normal('b2', mu=0, sigma=1.0, shape=(n_hidden,))

        # --- CAPA DE SALIDA (Entrada: n_hidden -> Salida: 1) ---
        sigma_wout = np.sqrt(1.0 / n_hidden)
        w_out = pm.Normal('w_out', mu=0, sigma=sigma_wout, shape=(n_hidden, 1))
        b_out = pm.Normal('b_out', mu=0, sigma=1.0, shape=(1,))

        # Propagación hacia adelante (Forward pass)
        act_1 = pm.math.maximum(0, pm.math.dot(X_shared, w1) + b1) 
        act_2 = pm.math.maximum(0, pm.math.dot(act_1, w2) + b2)    
        mu_out = pm.math.dot(act_2, w_out) + b_out 
        pm.Deterministic("mu_pred", mu_out.squeeze())

        # Modelado del Ruido del Sensor (Incertidumbre Aleatoria)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1.0)

        # Función de Verosimilitud
        y_obs = pm.Normal('y_obs', mu=mu_out.squeeze(), sigma=sigma_obs, observed=y_shared)

        # Inferencia
        print(f"Iniciando entrenamiento bayesiano mediante {inference_method.upper()}...")
        if inference_method == 'advi':
            # Estabilizamos ADVI usando el optimizador Adam con un learning rate controlado (0.002)
            # para evitar saltos destructivos en los parámetros variacionales
            approx = pm.fit(
                n=40000, 
                method='advi', 
                obj_optimizer=pm.adam(learning_rate=0.002), 
                progressbar=True
            ) 
            trace = approx.sample(draws=1000)
        elif inference_method == 'mcmc':
            trace = pm.sample(
                draws=1000,
                tune=2000,
                chains=2,         # Sube a 4 si tu CPU tiene 4+ núcleos
                cores=2,          # Sube a 4 si tu CPU tiene 4+ núcleos
                target_accept=0.95,
                return_inferencedata=True # Formato estándar y moderno
            )
            
    return bnn_model, trace

def predict_bnn(bnn_model, trace, X_test):
    """
    Distribución predictiva a posteriori para nuevos datos.
    """
    X_test_matrix = X_test[:, None]
    dummy_y = np.zeros(len(X_test)) 
    with bnn_model:
        pm.set_data({'X': X_test_matrix, 'y': dummy_y}) 
        posterior_predictive = pm.sample_posterior_predictive(trace) 
    return posterior_predictive