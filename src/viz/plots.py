import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, auc

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

def plot_mlp_decision_boundary(xx, yy, Z_mlp_entropy, X_train, y_train, X_ood):
    """
    Dibuja el mapa de calor de incertidumbre (Entropía Predictiva) para el MLP con MC Dropout.
    """
    print("\n--- Generando Mapa de Fronteras de Decisión (MLP) ---")
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap_scatter = ListedColormap(['#2ca02c', '#1f77b4', '#ff7f0e'])

    # CRÍTICO: Fijamos la misma escala absoluta de entropía para ambos mapas.
    levels = np.linspace(0.0, 1.1, 20)

    # PANEL A: MLP con MC Dropout
    contour = ax.contourf(xx, yy, Z_mlp_entropy, alpha=0.7, cmap='magma_r', levels=levels)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Entropía Predictiva (Incertidumbre)', fontsize=11)
    
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_scatter, edgecolors='k', s=30)
    ax.scatter(X_ood[:, 0], X_ood[:, 1], c='red', marker='X', s=80, edgecolors='k', label='Atacante (OOD)')
    
    ax.set_title('A: MLP (MC Dropout) - Incertidumbre Implícita', fontsize=13, fontweight='bold')
    ax.set_xlabel('Característica $X_1$')
    ax.set_ylabel('Característica $X_2$')
    ax.legend(loc='upper right')

    plt.tight_layout()
    
    # Guardado automático
    os.makedirs('reports/figures', exist_ok=True)
    filepath = 'reports/figures/fronteras_decision_mlp.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada en '{filepath}'")
    plt.show()

def plot_bnn_decision_boundary(xx, yy, Z_bnn_entropy, X_train, y_train, X_ood):
    """
    Dibuja el mapa de calor de incertidumbre (Entropía Predictiva) para la BNN.
    """
    print("\n--- Generando Mapa de Fronteras de Decisión (BNN) ---")
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap_scatter = ListedColormap(['#2ca02c', '#1f77b4', '#ff7f0e'])

    # CRÍTICO: Fijamos la misma escala absoluta de entropía para ambos mapas.
    levels = np.linspace(0.0, 1.1, 20)

    # PANEL B: BNN (PyMC)
    contour = ax.contourf(xx, yy, Z_bnn_entropy, alpha=0.7, cmap='magma_r', levels=levels)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Entropía Predictiva (Incertidumbre)', fontsize=11)
    
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_scatter, edgecolors='white', s=30)
    ax.scatter(X_ood[:, 0], X_ood[:, 1], c='red', marker='X', s=80, edgecolors='white', label='Atacante (OOD)')
    
    ax.set_title('B: BNN (PyMC) - Incertidumbre Explícita', fontsize=13, fontweight='bold')
    ax.set_xlabel('Característica $X_1$')
    ax.set_ylabel('Característica $X_2$') # Añadido el eje Y que antes no estaba
    ax.legend(loc='upper right')

    plt.tight_layout()
    
    # Guardado automático
    os.makedirs('reports/figures', exist_ok=True)
    filepath = 'reports/figures/fronteras_decision_bnn.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada en '{filepath}'")
    plt.show()

def plot_roc_ood_detection(entropy_mlp_id, entropy_mlp_ood, entropy_bnn_id, entropy_bnn_ood):
    """
    Dibuja la Curva ROC evaluando la capacidad de los modelos para detectar 
    ataques (Out-of-Distribution) basándose en su Entropía Predictiva.
    """
    print("\n--- Generando Curva ROC de Detección de Ataques ---")
    
    # Preparamos las etiquetas verdaderas: 0 para datos conocidos (ID), 1 para ataques (OOD)
    y_true = np.concatenate([np.zeros(len(entropy_mlp_id)), np.ones(len(entropy_mlp_ood))])
    
    # Juntamos las entropías (nuestras "puntuaciones de riesgo")
    scores_mlp = np.concatenate([entropy_mlp_id, entropy_mlp_ood])
    scores_bnn = np.concatenate([entropy_bnn_id, entropy_bnn_ood])
    
    # Calculamos ROC y AUC para MLP Clásico
    fpr_mlp, tpr_mlp, _ = roc_curve(y_true, scores_mlp)
    roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
    
    # Calculamos ROC y AUC para BNN
    fpr_bnn, tpr_bnn, _ = roc_curve(y_true, scores_bnn)
    roc_auc_bnn = auc(fpr_bnn, tpr_bnn)
    
    # --- Dibujamos la gráfica ---
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr_mlp, tpr_mlp, color='#1f77b4', lw=2.5, linestyle='--',
             label=f'MLP Clásico (AUC = {roc_auc_mlp:.3f})')
    
    plt.plot(fpr_bnn, tpr_bnn, color='#2ca02c', lw=2.5,
             label=f'BNN Probabilística (AUC = {roc_auc_bnn:.3f})')
    
    # Línea de azar (modelo inútil)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':', label='Azar (AUC = 0.500)')
    
    plt.title('Curva ROC: Detección de Ataques (OOD) usando Entropía', fontsize=14, fontweight='bold')
    plt.xlabel('Tasa de Falsos Positivos (Bloquear a usuarios legítimos)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (Bloquear al atacante)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Guardado automático
    os.makedirs('reports/figures', exist_ok=True)
    filepath = 'reports/figures/roc_ood_detection.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Curva ROC guardada en '{filepath}'")
    
    plt.show()



def plot_confidence_histogram(probs_mlp, probs_bnn, y_true):
    """
    Genera un histograma comparativo de las confianzas máximas emitidas por los modelos,
    separando los datos In-Distribution (ID) de los Out-of-Distribution (OOD).
    """
    print("\n--- Generando Histograma de Confianzas (ID vs OOD) ---")
    
    # Extraemos la confianza máxima de cada predicción
    conf_mlp = np.max(probs_mlp, axis=1)
    conf_bnn = np.max(probs_bnn, axis=1)
    
    # Separamos las confianzas según si el dato era legítimo (y >= 0) o atacante (y == -1)
    mask_id = y_true >= 0
    mask_ood = y_true == -1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # Configuraciones comunes
    bins = np.linspace(0.33, 1.0, 30)
    
    # Panel A: MLP Clásico
    axes[0].hist(conf_mlp[mask_id], bins=bins, alpha=0.6, color='blue', label='Usuarios Legítimos (ID)')
    axes[0].hist(conf_mlp[mask_ood], bins=bins, alpha=0.8, color='red', edgecolor='darkred', label='Atacante (OOD)')
    axes[0].set_title('A: MLP Clásico (Problema de Sobreconfianza)', fontweight='bold')
    axes[0].set_xlabel('Confianza Predicha')
    axes[0].set_ylabel('Número de Muestras')
    axes[0].legend(loc='upper center')
    axes[0].grid(True, alpha=0.3)
    
    # Panel B: BNN Probabilística
    axes[1].hist(conf_bnn[mask_id], bins=bins, alpha=0.6, color='green', label='Usuarios Legítimos (ID)')
    axes[1].hist(conf_bnn[mask_ood], bins=bins, alpha=0.8, color='red', edgecolor='darkred', label='Atacante (OOD)')
    axes[1].set_title('B: BNN Probabilística (Confianza Calibrada)', fontweight='bold')
    axes[1].set_xlabel('Confianza Predicha')
    axes[1].legend(loc='upper center')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('reports/figures', exist_ok=True)
    filepath = 'reports/figures/confidence_histogram.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Histograma de Confianzas guardado en '{filepath}'")
    
    plt.show()