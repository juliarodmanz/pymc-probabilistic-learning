import pymc as pm
import numpy as np

def construir_topologia_neuronal(X_tensor, n_hidden):
    """
    Función auxiliar que define la topología neuronal de la red.
    Al aislarla, podemos enchufarle tanto Minibatches como tensores pm.Data de la predicción.
    """
    n_features = X_tensor.shape[1] # Número de características
    
    # Capa Oculta
    w_in_hidden = pm.Normal('w_in_hidden', mu=0, sigma=10.0, shape=(n_features, n_hidden)) # Pesos de la capa oculta
    b_hidden = pm.Normal('b_hidden', mu=0, sigma=10.0, shape=(n_hidden,)) # Biases de la capa oculta
    act_hidden = pm.math.tanh(pm.math.dot(X_tensor, w_in_hidden) + b_hidden) # Activaciones de la capa oculta
    
    # Capa de Salida
    w_hidden_out = pm.Normal('w_hidden_out', mu=0, sigma=10.0, shape=(n_hidden,)) # Pesos de la capa de salida
    b_out = pm.Normal('b_out', mu=0, sigma=10.0) # Biases de la capa de salida
    
    mu_pred = pm.math.dot(act_hidden, w_hidden_out) + b_out # Media de la distribución a posteriori
    return mu_pred


def entrenar_bnn_advi(X_train, y_train, n_hidden=20, batch_size=256, n_iterations=100000, random_seed=2003):
    """
    Grafo de entrenamiento: Utiliza pm.Minibatch para optimizar el ELBO de forma escalable.
    """
    # Empaquetado estocástico, usamos pm.Minibatch para optimizar el ELBO
    X_mb = pm.Minibatch(X_train, batch_size=batch_size)
    y_mb = pm.Minibatch(y_train, batch_size=batch_size)
    
    with pm.Model() as bnn_train_model:
        # Inyectamos los minilotes en la topología neuronal
        mu_pred = construir_topologia_neuronal(X_mb, n_hidden)
        
        # Prior del ruido
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=5.0)
        
        # Verosimilitud escalada, ya que usamos pm.Minibatch 
        y_obs = pm.Normal('y_obs', mu=mu_pred, sigma=sigma_obs, observed=y_mb, total_size=X_train.shape[0])
        
        # Optimización
        mean_field = pm.fit(n=n_iterations, method='advi', obj_optimizer=pm.adam(learning_rate=0.001), random_seed=random_seed, progressbar=True) # Campo medio, usamos ADVI
        trace_vi = mean_field.sample(draws=1000, random_seed=random_seed) # Muestreo variacional
    return bnn_train_model, trace_vi, mean_field


def predict_bnn_advi(trace_vi, X_test, n_hidden=20):
    """
    Grafo Predictivo: Reconstruye la topología aislando la estocasticidad de los mini-batches.
    """
    dummy_y = np.zeros(len(X_test)) 
    
    with pm.Model() as bnn_predict_model:
        # Contenedores de memoria dinámica para la inyección de datos en la predicción
        X_data = pm.Data('X_data', X_test)
        y_data = pm.Data('y_data', dummy_y)
        
        # Inyectamos los datos limpios en la misma topología
        mu_pred = construir_topologia_neuronal(X_data, n_hidden)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=5.0)
        
        #  Verosimilitud predictiva
        y_obs = pm.Normal('y_obs', mu=mu_pred, sigma=sigma_obs, observed=y_data)
        
        #  Generación predictiva utilizando el conocimiento de la fase anterior
        post_pred = pm.sample_posterior_predictive(trace_vi)
        
    return post_pred