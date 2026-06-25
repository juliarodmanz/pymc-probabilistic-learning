import pymc as pm
import numpy as np
def entrenar_bnn_mcmc(X_train, y_train, n_hidden=20, random_seed=2003):
    """
    Construye y muestrea una Red Neuronal Bayesiana mediante MCMC (NUTS).
    Utiliza pm.Data para permitir inferencia Out-of-Sample mediante pm.set_data.
    """
    n_features = X_train.shape[1]
    
    with pm.Model() as bnn_model:
        # Instanciamos los contenedores dinámicos
        X_data = pm.Data('X_data', X_train)
        y_data = pm.Data('y_data', y_train)
        
        # Definimos la topología de la red
        w_in_hidden = pm.Normal('w_in_hidden', mu=0, sigma=10.0, shape=(n_features, n_hidden)) # Pesos de la capa oculta
        b_hidden = pm.Normal('b_hidden', mu=0, sigma=10.0, shape=(n_hidden,)) # Biases de la capa oculta
        act_hidden = pm.math.tanh(pm.math.dot(X_data, w_in_hidden) + b_hidden) # Activaciones de la capa oculta, tangente hiperbólica
        
        w_hidden_out = pm.Normal('w_hidden_out', mu=0, sigma=10.0, shape=(n_hidden,)) # Pesos de la capa de salida
        b_out = pm.Normal('b_out', mu=0, sigma=10.0) # Biases de la capa de salida
        mu_pred = pm.math.dot(act_hidden, w_hidden_out) + b_out # Media de la distribución a posteriori, sirve para la verosimilitud
        
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=5.0) # Prior del ruido

        y_obs = pm.Normal('y_obs', mu=mu_pred, sigma=sigma_obs, observed=y_data) # Verosimilitud
        
        # Muestreo NUTS
        trace_mcmc = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            target_accept=0.90,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True
        )
        
    return bnn_model, trace_mcmc

def predict_bnn_mcmc(bnn_model, trace_mcmc, X_test):
    """
    Genera el muestreo predictivo a posteriori para nuevos datos (Out-of-Sample)
    utilizando el motor de inferencia exacta MCMC.
    """
    # Vector estabilizador para cuadrar la dimensión de la matriz jacobiana
    dummy_y = np.zeros(len(X_test))
    
    with bnn_model:
        #. Inyección dinámica: Sustituimos el espacio de entrenamiento por el de test
        pm.set_data({
            'X_data': X_test, 
            'y_data': dummy_y
        })
        
        # Generación predictiva sobre la distribución a posteriori exacta
        post_pred = pm.sample_posterior_predictive(trace_mcmc)
        
    return post_pred