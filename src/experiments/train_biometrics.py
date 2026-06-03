import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.seeds import set_global_seed
from src.utils.metrics import calculate_predictive_entropy, calculate_mean_predictive_variance
from src.data.dataset_biometrics import generate_biometric_data
from src.models.mlp_classifier import BaselineClassifier, train_classifier, predict_mc_dropout
from src.models.bnn_classifier import build_and_train_bnn_classifier, predict_bnn_classifier
from src.viz.plots import plot_bnn_decision_boundary,plot_mlp_decision_boundary, plot_roc_ood_detection, plot_confidence_histogram

def main():
    """
    Experimento de Clasificación Biométrica con MLP Estocástico vs BNN
    Evaluamos ambos modelos frente a un atacante OOD y simulamos casos de uso en producción.
    """
    # Establecemos como semilla global 2003 para asegurar la reproducibilidad
    print("--- 0. Fijando semillas estocásticas (Reproducibilidad) ---")
    set_global_seed(2003)

    # Cargamos el dataset de biométricos
    print("\n--- 1. Cargando Dataset Biométrico ---")
    X_train, y_train, X_ood, y_ood = generate_biometric_data(n_samples=150)
    
    # Modelo Clásico Estocástico
    print("\n--- 2. Entrenando MLP Estocástico (MC Dropout + L2) ---")
    # Aumentamos las características para permitir fronteras cerradas
    X_train_aug = augment_features(X_train)
    # Entrenamos el modelo
    mlp_model = BaselineClassifier(input_dim=4, hidden_dim=32, output_dim=3) 
    mlp_model = train_classifier(mlp_model, X_train_aug, y_train, num_epochs=600, learning_rate=0.01)
    
    # Modelo Bayesiano
    print("\n--- 3. Entrenando BNN (PyMC + MCMC) ---")
    # Entrenamos el modelo
    bnn_model, trace = build_and_train_bnn_classifier(X_train_aug, y_train, n_hidden=20)

   # Evaluamos los modelos frente al Atacante (Out-of-Distribution)
    print("\n--- 4. EVALUACIÓN DE SEGURIDAD (Ataque OOD) ---")
    
    # Extraemos muestras estocásticas y medias de ambos modelos
    samples_mlp, probs_mlp_mean = predict_mc_dropout(mlp_model, augment_features(X_ood), num_samples=50)
    samples_bnn, probs_bnn_mean = predict_bnn_classifier(bnn_model, trace, augment_features(X_ood))

    # Calculamos Entropía
    entropy_mlp = calculate_predictive_entropy(probs_mlp_mean)
    entropy_bnn = calculate_predictive_entropy(probs_bnn_mean)
    
    # Calculamos Varianza Predictiva Media
    # Desviación típica a lo largo del eje de los muestreos
    var_mlp = calculate_mean_predictive_variance(samples_mlp.std(axis=0))
    var_bnn = calculate_mean_predictive_variance(samples_bnn.std(axis=0))

    # Reporte de Resultados de Autenticación
    
    print("\n================ RESULTADOS DE AUTENTICACIÓN ================")
    print("Métrica 1: Entropía Predictiva Media (Max: 1.0986)")
    print("Métrica 2: Varianza Predictiva Media (Dispersión OOD)")
    print("-------------------------------------------------------------")
    print(f"  MLP (MC Dropout): Entropía = {np.mean(entropy_mlp):.4f} | Varianza = {np.mean(var_mlp):.4f}")
    print(f"  BNN (MCMC):       Entropía = {np.mean(entropy_bnn):.4f} | Varianza = {np.mean(var_bnn):.4f}")
    print("=============================================================\n")

    print("\n--- 6. SIMULACIÓN DE CASOS DE USO EN PRODUCCIÓN ---")
    print("\n>>> CASO A: Intento legítimo (In-Distribution)")
    cara_legitima = [[1.5, 2.0]] 
    evaluate_new_face(mlp_model, bnn_model, trace, cara_legitima)

    print("\n>>> CASO B: Familiar sin permisos (In-Distribution)")
    cara_familiar = [[0.0, -1.5]]
    evaluate_new_face(mlp_model, bnn_model, trace, cara_familiar)

    print("\n>>> CASO C: Ataque desconocido (Out-of-Distribution)")
    cara_atacante = [[6.0, 3.0]] 
    evaluate_new_face(mlp_model, bnn_model, trace, cara_atacante)
    
   # Generamos los Mapas de Fronteras de Decisión
    print("\n--- 7. Calculando Mapeo de Incertidumbre Espacial ---")
    print("Muestreando la cuadrícula con ambos modelos (puede tardar unos segundos)...")
    
   # Establecemos el rango de la cuadrícula
    x_min, x_max = X_train[:, 0].min() - 1.5, X_ood[:, 0].max() + 1.5 
    y_min, y_max = X_ood[:, 1].min() - 1.5, X_train[:, 1].max() + 1.5
    
   # Generamos la cuadrícula
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60), np.linspace(y_min, y_max, 60))
    grid = np.c_[xx.ravel(), yy.ravel()]

   # Predicciones del MLP (MC Dropout) sobre el grid
    _, probs_mlp_grid = predict_mc_dropout(mlp_model, augment_features(grid), num_samples=30) #entrenamos con los datos augmentados del grid
    Z_mlp_entropy = calculate_predictive_entropy(probs_mlp_grid).reshape(xx.shape) # Hacemos reshape para que se ajuste al tamaño de la cuadrícula

   # Predicciones de la BNN sobre el grid
    _, probs_bnn_mean_grid = predict_bnn_classifier(bnn_model, trace, augment_features(grid)) #entrenamos con los datos augmentados del grid
    Z_bnn_entropy = calculate_predictive_entropy(probs_bnn_mean_grid).reshape(xx.shape) # Hacemos reshape para que se ajuste al tamaño de la cuadrícula

    plot_mlp_decision_boundary(xx, yy, Z_mlp_entropy, X_train, y_train, X_ood)
    plot_bnn_decision_boundary(xx, yy, Z_bnn_entropy, X_train, y_train, X_ood)

    # Curvas ROC y Calibración
    print("\n--- 8. Calculando métricas ROC para Detección de Anomalías ---")
    # Para la detección OOD, usamos la entropía como puntuación de anomalía (mayor entropía se traduce en más probable OOD)
    _, probs_mlp_id = predict_mc_dropout(mlp_model, X_train_aug, num_samples=50)
    entropy_mlp_id = calculate_predictive_entropy(probs_mlp_id)
    
    _, probs_bnn_mean_id = predict_bnn_classifier(bnn_model, trace, X_train_aug) 
    entropy_bnn_id = calculate_predictive_entropy(probs_bnn_mean_id)
    
    # Graficamos las curvas ROC para ambos modelos
    plot_roc_ood_detection(entropy_mlp_id, entropy_mlp, entropy_bnn_id, entropy_bnn)
    
    print("\n--- 9. Evaluando Calibración del Modelo ---")

    # Generamos un conjunto de validación con muestras legítimas e intrusas para evaluar la calibración de ambos modelos
    X_val, y_val, _, _ = generate_biometric_data(n_samples=200)
    X_calib = augment_features(np.vstack([X_val, X_ood]))
    y_calib = np.concatenate([y_val, np.full(len(X_ood), -1)])
    
    # Predicciones del MLP (MC Dropout) y de la BNN sobre el conjunto de validación
    _, probs_mlp_calib = predict_mc_dropout(mlp_model, X_calib, num_samples=50)
    _, probs_bnn_calib = predict_bnn_classifier(bnn_model, trace, X_calib)
    
    # Graficamos la calibración de ambos modelos usando histogramas de confianza para cada clase
    plot_confidence_histogram(probs_mlp_calib, probs_bnn_calib, y_calib)

def evaluate_new_face(mlp_model, bnn_model, trace, X_new, entropy_threshold=0.6):
    """
    Simula el sistema de seguridad en producción comparando ambos modelos.
    """
    X_new_array = np.array(X_new)
    if X_new_array.ndim == 1:
        X_new_array = X_new_array.reshape(1, -1)
        
    # PREDICCIÓN MLP (MC Dropout)
    X_new_aug = augment_features(X_new_array)
    # Obtenemos las muestras estocásticas y la media de probabilidades
    samples_mlp, probs_mlp_mean = predict_mc_dropout(mlp_model, X_new_aug, num_samples=50)
    # Calculamos la entropía y varianza predictiva para evaluar la incertidumbre
    entropy_mlp = calculate_predictive_entropy(probs_mlp_mean)[0] 
    var_mlp = calculate_mean_predictive_variance(samples_mlp.std(axis=0)) 
    # Clasificación y confianza del MLP
    class_mlp = np.argmax(probs_mlp_mean[0])
    conf_mlp = probs_mlp_mean[0][class_mlp]

    # PREDICCIÓN BNN (PyMC)
    # Obtenemos las muestras estocásticas y la media de probabilidades
    samples_bnn, p_mean_bnn = predict_bnn_classifier(bnn_model, trace, X_new_aug)
    # Calculamos la entropía y varianza predictiva para evaluar la incertidumbre
    entropy_bnn = calculate_predictive_entropy(p_mean_bnn)[0]
    var_bnn = calculate_mean_predictive_variance(samples_bnn.std(axis=0))
    # Clasificación y confianza de la BNN
    class_bnn = np.argmax(p_mean_bnn[0])
    conf_bnn = p_mean_bnn[0][class_bnn]
    
    # REPORTE VISUAL DE RESULTADOS
    print("\n" + "="*55)
    print(" INFORME DE AUTENTICACIÓN BIOMÉTRICA")
    print("="*55)
    print(f"Coordenadas faciales : {X_new_array[0]}")
    print("-" * 55)
    
    # MLP
    print("DECISIÓN MLP (MC Dropout):")
    print(f"  Probabilidades : [{probs_mlp_mean[0][0]:.3f}, {probs_mlp_mean[0][1]:.3f}, {probs_mlp_mean[0][2]:.3f}]")
    print(f"  Entropía       : {entropy_mlp:.4f}")
    print(f"  Varianza       : {var_mlp:.4f}")
    
    # Si la entropía es mayor que el umbral, se considera acceso denegado
    if entropy_mlp > entropy_threshold:
         print(" RESULTADO   : ACCESO DENEGADO (Incertidumbre detectada).")
    else:
         print(f" Predicción     : Clase {class_mlp} con {conf_mlp:.2%} de confianza.")
    
    print("-" * 55)
    
    # BNN
    print(" DECISIÓN BNN (PyMC):")
    print(f"  Probabilidades : [{p_mean_bnn[0][0]:.3f}, {p_mean_bnn[0][1]:.3f}, {p_mean_bnn[0][2]:.3f}]")
    print(f"  Entropía       : {entropy_bnn:.4f}")
    print(f"  Varianza       : {var_bnn:.4f}")
    
    # Si la entropía es mayor que el umbral, se considera acceso denegado
    if entropy_bnn > entropy_threshold:
        print(" RESULTADO   : ACCESO DENEGADO (Alta incertidumbre epistémica).")
    else:
        # Si la clase es 0, se considera acceso concedido, de lo contrario se deniega por ser un familiar sin permisos
        if class_bnn == 0:
            print(f"RESULTADO   : ACCESO CONCEDIDO (Usuario Autorizado) con {conf_bnn:.2%} de confianza.")
        else:
            print(f"RESULTADO   : ACCESO DENEGADO (Familiar {class_bnn} sin permisos).")
            
    print("="*55 + "\n")
def augment_features(X):
    """Eleva al cuadrado las características y las concatena (X -> X_aug)"""
    X_np = np.array(X)
    if X_np.ndim == 1:
        X_np = X_np.reshape(1, -1)
    return np.concatenate([X_np, X_np**2], axis=1)
if __name__ == '__main__':
    main()

