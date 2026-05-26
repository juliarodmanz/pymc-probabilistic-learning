import matplotlib.pyplot as plt
import os
import numpy as np

def plot_robot_comparison(X_train, y_train, X_test, y_true, y_pred_mlp, y_pred_bnn_mean, y_pred_bnn_std):
    """
    Genera y guarda la gráfica comparativa entre el modelo determinista y el bayesiano.
    """
    print("\n--- Generando Gráfica Comparativa ---")
    plt.figure(figsize=(12, 6))
    
    # 1. Graficamos el Ground Truth y los datos de entrenamiento
    plt.plot(X_test, y_true, 'k--', label='Función Real (Ground Truth)', alpha=0.7)
    plt.scatter(X_train, y_train, c='red', alpha=0.5, label='Datos de Entrenamiento (Sensor Activo)')
    
    # 2. Graficamos la predicción del MLP
    plt.plot(X_test, y_pred_mlp, 'b-', linewidth=2, label='Predicción MLP Determinista')
    
    # 3. Graficamos la predicción de la BNN y su incertidumbre
    plt.plot(X_test, y_pred_bnn_mean, 'g-', linewidth=2, label='Media Predictiva BNN')
    plt.fill_between(X_test, 
                     y_pred_bnn_mean - 1.96*y_pred_bnn_std, 
                     y_pred_bnn_mean + 1.96*y_pred_bnn_std, 
                     color='green', alpha=0.2, label='Incertidumbre Predictiva (95% CI)')
    
    # Zona de fallo del sensor sombreada
    plt.axvspan(4, 7, color='gray', alpha=0.1, label='Fallo del Sensor (Zona Ciega)')
    
    plt.title('Comparativa de Modelos frente a Vacíos Epistémicos', fontsize=14)
    plt.xlabel('Tiempo / Posición (x)', fontsize=12)
    plt.ylabel('Señal Cinemática (y)', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    # Guardamos la imagen
    os.makedirs('reports/figures', exist_ok=True)
    filepath = 'reports/figures/comparativa_robot.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada en '{filepath}'")
    
    plt.show()

def plot_uncertainty_decomposition(X_test, y_pred_bnn_std, sigma_aleatoria_std):
    """
    Grafica la evolución espacial de las componentes de la incertidumbre.
    Aísla la magnitud de la duda del modelo frente a la posición.
    """
    print("\n--- Generando Gráfica de Descomposición de Incertidumbre ---")
    plt.figure(figsize=(10, 5))
    
    # 1. Cálculos de varianzas y desviaciones para TODOS los puntos (500)
    var_total = y_pred_bnn_std**2
    var_aleatoria = sigma_aleatoria_std**2
    var_epistemica = np.maximum(var_total - var_aleatoria, 0.0)
    
    std_epistemica = np.sqrt(var_epistemica)
    std_total = y_pred_bnn_std
    
    # 2. Graficamos las tres magnitudes (nota la 'r' delante de los strings de las etiquetas)
    plt.plot(X_test, std_total, 'g-', linewidth=2, label=r'Incertidumbre Total ($\sigma_{total}$)')
    plt.plot(X_test, std_epistemica, 'purple', linestyle='--', linewidth=2.5, label=r'Incertidumbre Epistémica ($\sigma_{epist}$)')
    plt.axhline(y=sigma_aleatoria_std, color='orange', linestyle='-.', linewidth=2, label=r'Incertidumbre Aleatoria ($\sigma_{aleat}$)')
    
    # 3. Sombreado de la zona ciega
    plt.axvspan(4, 7, color='gray', alpha=0.1, label='Fallo del Sensor (Zona Ciega)')
    
    # Formato y estilo
    plt.title('Descomposición Espacial de la Incertidumbre Bayesiana', fontsize=14)
    plt.xlabel('Tiempo / Posición (x)', fontsize=12)
    plt.ylabel(r'Magnitud de la Incertidumbre ($\sigma$)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardado
    os.makedirs('reports/figures', exist_ok=True)
    filepath = 'reports/figures/descomposicion_incertidumbre.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada en '{filepath}'")
    
    plt.show()