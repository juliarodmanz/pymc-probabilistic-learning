import pymc as pm
import pytensor.tensor as pt
import numpy as np

def build_and_train_bnn(X_train, y_train, n_hidden=50, inference_method='advi'):
    """
    Construcción y entrenamiento de una Red Neuronal Bayesiana usando PyMC.
    La arquitectura es idéntica al modelo baseline (50x50, ReLU).
    """
    # Convertimos las entradas a matrices de columna (N, 1) para PyTensor, ya que PyMC espera entradas en formato matricial para operaciones de dot product
    X_train_matrix = X_train[:, None]
    
    with pm.Model() as bnn_model:
        # Datos compartidos (necesario para poder inyectar X_test después, ya que si no lo hacemos los datos de entrenamiento quedarían "fijados" en el modelo y no podríamos usarlo para predecir con nuevos datos)
        X_shared = pm.Data('X', X_train_matrix)
        y_shared = pm.Data('y', y_train)

        # Distribuciones a Priori de los Pesos y Sesgos (Incertidumbre Epistémica)
        # Capa Oculta 1
        w1 = pm.Normal('w1', mu=0, sigma=1, shape=(1, n_hidden))
        b1 = pm.Normal('b1', mu=0, sigma=1, shape=(n_hidden,))

        # Capa Oculta 2
        w2 = pm.Normal('w2', mu=0, sigma=1, shape=(n_hidden, n_hidden))
        b2 = pm.Normal('b2', mu=0, sigma=1, shape=(n_hidden,))

        # Capa de Salida
        w_out = pm.Normal('w_out', mu=0, sigma=1, shape=(n_hidden, 1))
        b_out = pm.Normal('b_out', mu=0, sigma=1, shape=(1,))

        # Propagación hacia adelante con operaciones tensoriales (PyTensor) y funciones de activación ReLU)
        act_1 = pm.math.maximum(0, pm.math.dot(X_shared, w1) + b1) # Función ReLU
        act_2 = pm.math.maximum(0, pm.math.dot(act_1, w2) + b2)    # Función ReLU
        mu_out = pm.math.dot(act_2, w_out) + b_out # Salida sin activación (regresión)

        # Modelado del Ruido del Sensor (Incertidumbre Aleatoria)
        # Usamos HalfNormal porque la desviación estándar siempre es positiva
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)

        # Función de Verosimilitud (Probabilidad de los datos dados los parámetros del modelo)
        y_obs = pm.Normal('y_obs', mu=mu_out.squeeze(), sigma=sigma_obs, observed=y_shared)

        # Inferencia: Estimación de la distribución a posteriori
        print(f"Iniciando entrenamiento bayesiano mediante {inference_method.upper()}...")
        if inference_method == 'advi': # Auto-Differentiation Variational Inference (ADVI) es un método de inferencia variacional que optimiza la aproximación a la distribución a posteriori. Es más rápido que MCMC y escala mejor con modelos complejos.
            approx = pm.fit(n=30000, method='advi', progressbar=True) # Ajustamos el modelo usando ADVI, el número de iteraciones puede ser ajustado según la complejidad del modelo y el tamaño del dataset
            trace = approx.sample(draws=1000) # Muestreamos la aproximación a la distribución a posteriori para obtener muestras de los parámetros del modelo
        elif inference_method == 'mcmc': # Monte Carlo por Cadenas de Markov (MCMC) es un método de inferencia que genera muestras de la distribución a posteriori utilizando cadenas de Markov. Es más lento que ADVI pero puede proporcionar una convergencia exacta a la distribución a posteriori.
            trace = pm.sample(draws=1000, tune=1000, cores=2, target_accept=0.9) # Muestreamos la distribución a posteriori usando MCMC, el número de muestras y de afinamiento (tune) puede ser ajustado según la complejidad del modelo y el tamaño del dataset
            
    return bnn_model, trace # Devolvemos el modelo y la traza de la inferencia para poder usarla posteriormente en la generación de predicciones con nuevos datos (X_test)

def predict_bnn(bnn_model, trace, X_test):
    """
    Distribución predictiva a posteriori para nuevos datos.
    """
    X_test_matrix = X_test[:, None]
    
    with bnn_model:
        # Sustituimos los datos de entrenamiento por los de validación
        pm.set_data({'X': X_test_matrix})
        # Muestreamos la red ensamblada para predecir
        posterior_predictive = pm.sample_posterior_predictive(trace) #Usamos la traza de la inferencia para generar muestras de la distribución predictiva a posteriori, lo que nos permite obtener no solo una predicción puntual sino también una distribución de posibles valores para cada punto de prueba, reflejando así la incertidumbre en las predicciones
        
    return posterior_predictive # Devolvemos la distribución predictiva a posteriori, que contiene muestras de las predicciones para cada punto de prueba, lo que nos permite analizar la incertidumbre en las predicciones y obtener intervalos de confianza o realizar análisis de sensibilidad según sea necesario.