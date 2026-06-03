import pymc as pm
import numpy as np

def build_and_train_bnn_classifier(X_train, y_train, n_hidden=20,inference_method='mcmc'):
    """
    Construcción y entrenamiento de una Red Neuronal Bayesiana para Clasificación multiclase.
    """
    # Número de características y clases
    n_features = X_train.shape[1] 
    n_classes = len(np.unique(y_train))
    
    # Definimos el modelo
    with pm.Model() as bnn_model:
        # Contenedores de datos compartidos, que permiten alimentar el modelo con diferentes conjuntos 
        X_shared = pm.Data('X', X_train)
        y_shared = pm.Data('y', y_train)

        # CAPA OCULTA 
        sigma_w1 = np.sqrt(2.0 / n_features) # Escalado de He/Xavier para la capa oculta, adaptado a la dimensionalidad de entrada
        w1 = pm.Normal('w1', mu=0, sigma=sigma_w1, shape=(n_features, n_hidden)) 
        b1 = pm.Normal('b1', mu=0, sigma=1.0, shape=(n_hidden,)) 

        # CAPA DE SALIDA 
        sigma_wout = np.sqrt(1.0 / n_hidden) # Escalado de He/Xavier para la capa de salida, adaptado a la cantidad de neuronas en la capa oculta
        w_out = pm.Normal('w_out', mu=0, sigma=sigma_wout, shape=(n_hidden, n_classes))
        b_out = pm.Normal('b_out', mu=0, sigma=1.0, shape=(n_classes,))

        # Propagación hacia adelante 
        act_1 = pm.math.maximum(0, pm.math.dot(X_shared, w1) + b1) # ReLU como función de activación para la capa oculta 
        logits = pm.math.dot(act_1, w_out) + b_out # Logits sin activar, que serán transformados por Softmax en la siguiente etapa
        
        # Activación Final: Softmax probabilístico
        # Transforma los logits espaciales en un vector de probabilidades que suma 1
        p = pm.math.softmax(logits, axis=-1)
        pm.Deterministic("p_pred", p) # Guardamos las probabilidades estimadas

        # Función de Verosimilitud para clasificación
        y_obs = pm.Categorical('y_obs', p=p, observed=y_shared)

        # Inferencia
        print(f"Iniciando entrenamiento bayesiano mediante {inference_method.upper()}...")
        if inference_method == 'advi':
            # Estabilizamos ADVI usando el optimizador Adam con un learning rate controlado (0.002)
            # para evitar saltos destructivos en los parámetros variacionales
            approx = pm.fit(
                n=40000, # Número de iteraciones aumentado para mejorar la convergencia en el espacio de clases
                method='advi', 
                obj_optimizer=pm.adam(learning_rate=0.002), 
                progressbar=True
            ) 
            trace = approx.sample(draws=1000)
        elif inference_method == 'mcmc':
            trace = pm.sample(
                draws=1000, # Número de muestras posterior aumentado para mejorar la exploración del espacio de clases
                tune=2000,  # Calentamiento largo para explorar bien el espacio de clases
                chains=2,   # 2 cadenas para balancear entre exploración y tiempo de cómputo, ajusta según tu CPU
                cores=2,    # Utiliza 2 núcleos para acelerar el muestreo, ajusta según tu CPU
                target_accept=0.95, # Cautela alta para evitar divergencias en la topología Softmax
                return_inferencedata=True # Retorna un objeto InferenceData para facilitar el análisis posterior
            )
            
    return bnn_model, trace

def predict_bnn_classifier(bnn_model, trace, X_test):
    """
    Calcula la distribución predictiva a posteriori para nuevas muestras
    Retorna el array de probabilidades y la media.
    """
    # Preparación de datos de prueba para el modelo, con etiquetas dummy ya que no se usarán en la predicción
    dummy_y = np.zeros(len(X_test), dtype=int) 
    with bnn_model:
        # Asignamos los datos de prueba
        pm.set_data({'X': X_test, 'y': dummy_y}) 
        # Muestreamos las predicciones posteriores
        posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["p_pred"]) 
    
    # La forma típica será (cadenas, draws, muestras, clases)
    p_samples = posterior_predictive.posterior_predictive['p_pred'].values # Extraemos el array de probabilidades
    p_samples = p_samples.reshape(-1, p_samples.shape[-2], p_samples.shape[-1]) # Colapsamos cadenas y draws para obtener un array de forma (total_muestras_mcmc, muestras_test, clases)
    
    # Probabilidad predictiva media
    p_mean = p_samples.mean(axis=0)
    
    return p_samples, p_mean