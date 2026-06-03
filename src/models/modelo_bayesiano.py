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

        # CAPA OCULTA 1
        w1 = pm.Normal('w1', mu=0, sigma=1.0, shape=(1, n_hidden)) 
        b1 = pm.Normal('b1', mu=0, sigma=1.0, shape=(n_hidden,))

        # CAPA OCULTA 2 
        # Escalamos la incertidumbre inicial según las neuronas de la capa anterior para evitar saturación de gradientes
        sigma_w2 = np.sqrt(2.0 / n_hidden) 
        w2 = pm.Normal('w2', mu=0, sigma=sigma_w2, shape=(n_hidden, n_hidden))
        b2 = pm.Normal('b2', mu=0, sigma=1.0, shape=(n_hidden,))

        # CAPA DE SALIDA 
        sigma_wout = np.sqrt(1.0 / n_hidden) # Escalamos la incertidumbre inicial según las neuronas de la capa anterior para evitar saturación de gradientes
        w_out = pm.Normal('w_out', mu=0, sigma=sigma_wout, shape=(n_hidden, 1))
        b_out = pm.Normal('b_out', mu=0, sigma=1.0, shape=(1,))

        # Propagación hacia adelante con activaciones ReLU
        act_1 = pm.math.maximum(0, pm.math.dot(X_shared, w1) + b1) # Primera capa oculta con ReLU
        act_2 = pm.math.maximum(0, pm.math.dot(act_1, w2) + b2)    # Segunda capa oculta con ReLU
        mu_out = pm.math.dot(act_2, w_out) + b_out # Salida sin activación para regresión
        pm.Deterministic("mu_pred", mu_out.squeeze()) # Predicción a posteriori

        # Modelado del Ruido del Sensor (Incertidumbre Aleatoria)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1.0) #Sigue un Semigaussiana para garantizar positividad y permitir una amplia gama de valores, reflejando la incertidumbre inherente en las mediciones del sensor.

        # Función de Verosimilitud
        y_obs = pm.Normal('y_obs', mu=mu_out.squeeze(), sigma=sigma_obs, observed=y_shared)

        # Inferencia
        print(f"Iniciando entrenamiento bayesiano mediante {inference_method.upper()}...")
        if inference_method == 'advi':
            # Estabilizamos ADVI usando el optimizador Adam con un learning rate controlado (0.002)
            # para evitar saltos destructivos en los parámetros variacionales
            approx = pm.fit(
                n=40000, # Aumentamos el número de iteraciones para permitir una mejor convergencia, especialmente con un learning rate más bajo
                method='advi', 
                obj_optimizer=pm.adam(learning_rate=0.002), 
                progressbar=True
            ) 
            trace = approx.sample(draws=1000)
        elif inference_method == 'mcmc':
            trace = pm.sample(
                draws=1000, # Número de muestras a recolectar
                tune=2000, # Calentamiento para permitir que la cadena alcance la región de alta probabilidad antes de recolectar muestras
                chains=2, # Número de cadenas MCMC para evaluar la convergencia (idealmente 4, pero ajustamos a 2 para reducir el tiempo de cómputo)
                cores=2,  # Número de cores para acelerar la ejecución en paralelo
                target_accept=0.95, # Aumentamos el target_accept para reducir la tasa de rechazo y mejorar la exploración del espacio de parámetros, especialmente en modelos complejos
                return_inferencedata=True # Devolvemos un objeto InferenceData para facilitar el análisis posterior
            )
            
    return bnn_model, trace

def predict_bnn(bnn_model, trace, X_test):
    """
    Distribución predictiva a posteriori para nuevos datos.
    """
    X_test_matrix = X_test[:, None]
    # Preparación de datos de prueba para el modelo, con etiquetas dummy ya que no se usarán en la predicción
    dummy_y = np.zeros(len(X_test)) 
    with bnn_model:
        # Asignamos los datos de prueba
        pm.set_data({'X': X_test_matrix, 'y': dummy_y}) 
        # Muestreamos las predicciones posteriores
        posterior_predictive = pm.sample_posterior_predictive(trace) 
    return posterior_predictive