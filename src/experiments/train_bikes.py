import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.bnn_bike_advi import entrenar_bnn_advi, predict_bnn_advi
from src.data.dataset_bikes import cargar_datos_diarios, cargar_datos_horarios
from src.utils.seeds import set_global_seed
from src.models.bnn_bike_mcmc import entrenar_bnn_mcmc,predict_bnn_mcmc
from src.utils.metrics import rmse, calculate_coverage_95, calculate_mean_predictive_variance
from src.viz.plots import plot_bikes_mcmc_vs_advi,plot_elbo
def main():
    # Establecemos una semilla global para asegurar la reproducibilidad de los resultados, lo que es crucial en experimentos científicos para poder comparar resultados y validar conclusiones.
    print("--- 0. Fijando semillas estocásticas (Reproducibilidad) ---")
    set_global_seed(2003)  


    print("\n--- 1.1. Cargando Conjunto de Datos Diarios ---")
    X_train_day, X_test_day, y_train_day, y_test_day, scaler_X_day, scaler_y_day = cargar_datos_diarios(random_state=2003) #cargamos el dataset de datos diarios
    
    print("\n--- 1.2. Cargando Conjunto de Datos Horarios ---")
    X_train_hour, X_test_hour, y_train_hour, y_test_hour, scaler_X_hour, scaler_y_hour = cargar_datos_horarios(random_state=2003) #cargamos el dataset de datos horarios
     
    # Fase 1: 

    print("\n--- 2.1. [Fase 1] Entrenando con ADVI (day.csv)... ---")
    modelo_dia_advi, traza_dia_advi, mean_field_dia = entrenar_bnn_advi(X_train_day, y_train_day, batch_size=64, n_iterations=30000, random_seed=2003) #entrenamos el modelo con ADVI
    
    print("\n--- 2.2. [Fase 1] Entrenando con MCMC / NUTS (day.csv)... ---")
    modelo_dia_mcmc, traza_dia_mcmc = entrenar_bnn_mcmc(X_train_day, y_train_day, n_hidden=20, random_seed=2003) #entrenamos el modelo con MCMC
    
    
    print("\n[Generando Predicciones Out-of-Sample - Fase 1]")
    pred_dia_advi = predict_bnn_advi(traza_dia_advi, X_test_day, n_hidden=20) #generamos predicciones con ADVI
    pred_dia_mcmc = predict_bnn_mcmc(modelo_dia_mcmc, traza_dia_mcmc, X_test_day) #generamos predicciones con MCMC

    # Evaluación Comparativa Fase 1
    rmse_advi_1, cob_advi_1, var_advi_1, mean_advi_1, std_advi_1, y_test_real = procesar_predicciones(pred_dia_advi, y_test_day, scaler_y_day)
    rmse_mcmc_1, cob_mcmc_1, var_mcmc_1,mean_mcmc_1, std_mcmc_1, _ = procesar_predicciones(pred_dia_mcmc, y_test_day, scaler_y_day)

    print("\n--- RESULTADOS FASE 1 ---")
    print(f"MCMC (Inferencia Exacta):")
    print(f"  - RMSE: {rmse_mcmc_1:.2f}")  
    print(f"  - Cobertura 95% (PICP): {cob_mcmc_1:.2f}%")
    print(f"  - Varianza Predictiva: {var_mcmc_1:.2f}")
    
    print(f"\nADVI (Campo Medio):")
    print(f"  - RMSE: {rmse_advi_1:.2f}")
    print(f"  - Cobertura 95% (PICP): {cob_advi_1:.2f}%")
    print(f"  - Varianza Predictiva: {var_advi_1:.2f}")

    # Fase 2: Escalabilidad con ADVI
    print("\n" + "="*50)
    print(" FASE 2: ESCALABILIDAD MASIVA CON ADVI")
    print("="*50)
    
    print("\n[Entrenando ADVI Estocástico - Datos Horarios]")
    modelo_hora_advi, traza_hora_advi, mean_field_hora = entrenar_bnn_advi(X_train_hour, y_train_hour, batch_size=256, n_iterations=100000, random_seed=2003) #entrenamos el modelo con ADVI
    
    print("\n[Generando Predicciones Out-of-Sample - Fase 2]")
    pred_hora_advi = predict_bnn_advi(traza_hora_advi, X_test_hour, n_hidden=20) #generamos predicciones con ADVI
    
    # Evaluación Fase 2
    rmse_advi_2, cob_advi_2, var_advi_2,_,_,_ = procesar_predicciones(pred_hora_advi, y_test_hour, scaler_y_hour)
    
    print("\n--- RESULTADOS FASE 2 ---")
    print(f"ADVI Horario (Big Data):")
    print(f"  - RMSE: {rmse_advi_2:.2f}")
    print(f"  - Cobertura 95% (PICP): {cob_advi_2:.2f}%")
    print(f"  - Varianza Predictiva: {var_advi_2:.2f}")

    # Gráfica de convergencia final
    plot_bikes_mcmc_vs_advi(y_true=y_test_real, y_mean_mcmc=mean_mcmc_1, y_std_mcmc=std_mcmc_1, y_mean_advi=mean_advi_1, y_std_advi=std_advi_1,num_points=100)
    
    # Gráfica de elbo fase 2
    plot_elbo(mean_field_hora)
    
def procesar_predicciones(post_pred, y_test_scaled, scaler_y):
    """
    Función auxiliar para desnormalizar las predicciones y calcular métricas.
    Actualizada para retornar los tensores necesarios para la visualización.
    """
    # Extracción de la matriz de predicciones colapsando cadenas
    pred_flat = post_pred.posterior_predictive['y_obs'].values
    pred_flat = pred_flat.reshape(-1, pred_flat.shape[-1])
    
    # Desnormalización a escala real (bicicletas)
    pred_real = scaler_y.inverse_transform(pred_flat.T).T 
    pred_real = np.maximum(0, pred_real) # Límite físico: no hay bicis negativas
    y_test_real = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    # Cálculo de momentos estadísticos
    y_pred_mean = np.mean(pred_real, axis=0)
    y_pred_std = np.std(pred_real, axis=0)
    
    # Cálculo de métricas
    error_rmse = rmse(y_test_real, y_pred_mean)
    cobertura = calculate_coverage_95(y_test_real, y_pred_mean, y_pred_std)
    var_predictiva = calculate_mean_predictive_variance(y_pred_std)
    
    # Retornamos las métricas numéricas + los tensores para la gráfica
    return error_rmse, cobertura, var_predictiva, y_pred_mean, y_pred_std, y_test_real

if __name__ == '__main__':
    main()