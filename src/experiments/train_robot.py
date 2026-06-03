import numpy as np
import sys
import os

# Apuntamos a la raíz del proyecto (dos carpetas hacia atrás: experiments -> src -> raíz)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Ahora importamos indicando que todo nace de la carpeta 'src'
from src.utils.seeds import set_global_seed
from src.utils.metrics import rmse, decompose_uncertainty, calculate_coverage_95, calculate_mean_predictive_variance
from src.data.dataset_robot import generate_robot_data
from src.models.mlp_determinista import BaselineNN, train_model, predict_model
from src.models.modelo_bayesiano import build_and_train_bnn, predict_bnn
from src.viz.plots import plot_robot_comparison, plot_uncertainty_decomposition

def main():
    # Establecemos una semilla global para asegurar la reproducibilidad de los resultados, lo que es crucial en experimentos científicos para poder comparar resultados y validar conclusiones.
    print("--- 0. Fijando semillas estocásticas (Reproducibilidad) ---")
    set_global_seed(2003)  

    print("\n--- 1. Cargando Conjunto de Datos Sintético ---")
    X_train, y_train, X_test, y_true = generate_robot_data()
    X_test = np.sort(X_test) 

    # Normalizamos los datos para la BNN, pero no para el MLP clásico, ya que PyTorch es robusto a la escala de los datos y queremos mantenerlo lo más simple posible para el modelo determinista. 
    # La normalización es especialmente importante para la BNN para asegurar una convergencia estable durante el entrenamiento bayesiano, evitando problemas de escalado que pueden afectar la calidad de las inferencias posteriores.
    X_mean, X_std = X_train.mean(), X_train.std()  
    y_mean, y_std = y_train.mean(), y_train.std()

    X_train_norm = (X_train - X_mean) / X_std 
    X_test_norm = (X_test - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std

    # PyTorch es robusto: le pasamos los datos originales sin normalizar haciendo reshaope para que sean compatibles con la entrada de la red neuronal.
    X_train_pt = X_train.reshape(-1, 1)
    y_train_pt = y_train.reshape(-1, 1)
    X_test_pt = X_test.reshape(-1, 1)

    print("\n--- 2. Entrenando Modelo Clásico Determinista (PyTorch) ---")

    initial_mlp_model = BaselineNN(input_dim=1) # Arquitectura con una sola entrada (posición del robot) y una salida (distancia al objetivo)
    mlp_model = train_model(initial_mlp_model, X_train_pt, y_train_pt, num_epochs=1500, learning_rate=0.01) # Aumentamos el número de épocas para permitir una mejor convergencia, especialmente dado que no estamos normalizando los datos para el MLP clásico

    y_pred_mlp = predict_model(mlp_model, X_test_pt)
    y_pred_mlp = y_pred_mlp.flatten() # Aplanamos las predicciones para que tengan la misma forma que y_true, facilitando el cálculo de métricas y la visualización posterior

    print("\n--- 3. Entrenando Red Neuronal Bayesiana (PyMC) ---")
    # Entrenamos la BNN usando X e Y normalizados
    bnn_model, trace = build_and_train_bnn(X_train_norm, y_train_norm, n_hidden=50, inference_method='mcmc')

    # Generamos predicciones pasándole la X normalizada
    posterior_pred = predict_bnn(bnn_model, trace, X_test_norm)
    y_bnn_samples_norm = posterior_pred.posterior_predictive['y_obs'].values.reshape(-1, len(X_test)) # Obtenemos las muestras de la predicción a posteriori para cada punto de prueba, con forma (n_samples, n_test_points)

    # Deshacemos la normalización multiplicando por la desviación estándar original
    # Esto restaura tanto la media de la predicción como la escala de su incertidumbre
    y_bnn_samples = (y_bnn_samples_norm * y_std) + y_mean
    y_pred_bnn_mean = y_bnn_samples.mean(axis=0)
    y_pred_bnn_std = y_bnn_samples.std(axis=0)

    # CÁLCULO DE MÉTRICAS Y VISUALIZACIONES
    mask_obs = (X_test < 4) | (X_test > 7) # Máscara para evaluar solo en la zona observada, excluyendo la zona ciega
    mask_blind = (X_test >= 4) & (X_test <= 7) # Máscara para evaluar solo en la zona ciega, donde no hay datos de entrenamiento

    print("\n================ RESULTADOS DE LA EVALUACIÓN ================")
    print("ZONA OBSERVABLE (In-Distribution):")
    print(f"  RMSE MLP Clásico:  {rmse(y_true[mask_obs], y_pred_mlp[mask_obs]):.4f}")
    print(f"  RMSE BNN:          {rmse(y_true[mask_obs], y_pred_bnn_mean[mask_obs]):.4f}")
    print(f"  PICP BNN (95%):    {calculate_coverage_95(y_true[mask_obs], y_pred_bnn_mean[mask_obs], y_pred_bnn_std[mask_obs]):.2f}%")

    print("\nZONA CIEGA (Extrapolación OOD - Fallo Sensor):")
    print("  Varianza Predictiva Media (MPV):")
    print("    MLP Clásico:     0.0000 (Exceso de certeza)")
    print(f"    BNN:             {calculate_mean_predictive_variance(y_pred_bnn_std[mask_blind]):.4f} (Detecta la ignorancia)")
    print("=============================================================\n")
    
    sigma_obs_aprendido_norm = float(trace.posterior['sigma_obs'].mean()) # Obtenemos la incertidumbre aleatoria aprendida por la BNN en su escala normalizada
    sigma_aleatoria_std = sigma_obs_aprendido_norm * y_std # Restauramos la incertidumbre aleatoria en su escala original
    
    # 1. Gráfica de trayectorias 
    plot_robot_comparison(X_train, y_train, X_test, y_true, y_pred_mlp, y_pred_bnn_mean, y_pred_bnn_std)

    # 2. Gráfica analítica de la duda 
    plot_uncertainty_decomposition(X_test, y_pred_bnn_std, sigma_aleatoria_std)
    
if __name__ == '__main__':
    main()