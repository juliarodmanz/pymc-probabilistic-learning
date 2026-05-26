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

    # === EL TRUCO MATEMÁTICO: NORMALIZACIÓN COMPLETA (Z-SCORE) ===
    X_mean, X_std = X_train.mean(), X_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()

    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std

    # PyTorch es robusto: le pasamos los datos originales sin normalizar para no tocar lo que funciona
    X_train_pt = X_train.reshape(-1, 1)
    y_train_pt = y_train.reshape(-1, 1)
    X_test_pt = X_test.reshape(-1, 1)

    print("\n--- 2. Entrenando Modelo Clásico Determinista (PyTorch) ---")
    initial_mlp_model = BaselineNN(input_dim=1)
    mlp_model = train_model(initial_mlp_model, X_train_pt, y_train_pt, num_epochs=1500, learning_rate=0.01)

    y_pred_mlp = predict_model(mlp_model, X_test_pt)
    y_pred_mlp = y_pred_mlp.flatten()

    print("\n--- 3. Entrenando Red Neuronal Bayesiana (PyMC) ---")
    # 1. Entrenamos la BNN usando X e Y normalizados
    bnn_model, trace = build_and_train_bnn(X_train_norm, y_train_norm, n_hidden=50, inference_method='mcmc')

    # 2. Generamos predicciones pasándole la X normalizada
    posterior_pred = predict_bnn(bnn_model, trace, X_test_norm)
    y_bnn_samples_norm = posterior_pred.posterior_predictive['y_obs'].values.reshape(-1, len(X_test))

    # 3. ¡LA CLAVE! Deshacemos la normalización multiplicando por la desviación estándar original
    # Esto restaura tanto la media de la predicción como la escala de su incertidumbre
    y_bnn_samples = (y_bnn_samples_norm * y_std) + y_mean
    y_pred_bnn_mean = y_bnn_samples.mean(axis=0)
    y_pred_bnn_std = y_bnn_samples.std(axis=0)

    # --- CÁLCULO DE MÉTRICAS ---
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
    
    sigma_obs_aprendido_norm = float(trace.posterior['sigma_obs'].mean())
    sigma_aleatoria_std = sigma_obs_aprendido_norm * y_std
    
    # 1. Gráfica de trayectorias (Figura 1)
    plot_robot_comparison(X_train, y_train, X_test, y_true, y_pred_mlp, y_pred_bnn_mean, y_pred_bnn_std)

    # 2. Gráfica analítica de la duda (Figura 2)
    plot_uncertainty_decomposition(X_test, y_pred_bnn_std, sigma_aleatoria_std)
    
if __name__ == '__main__':
    main()