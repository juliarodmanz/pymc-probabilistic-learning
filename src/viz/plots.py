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
    plt.figure(figsize=(12, 6)) # Ajustamos el tamaño de la gráfica
    plt.plot(X_test, y_true, 'k--', label='Función Real (Ground Truth)', alpha=0.7) # Graficamos el Ground Truth (la función verdadera)
    plt.scatter(X_train, y_train, c='red', alpha=0.5, label='Datos de Entrenamiento (Sensor Activo)') #Dibujamos los datos de entrenamiento
    
    plt.plot(X_test, y_pred_mlp, 'b-', linewidth=2, label='Predicción MLP Determinista') # Graficamos la predicción del MLP
    
    plt.plot(X_test, y_pred_bnn_mean, 'g-', linewidth=2, label='Media Predictiva BNN') # Graficamos la media predictiva
    plt.fill_between(X_test, 
                     y_pred_bnn_mean - 1.96*y_pred_bnn_std, 
                     y_pred_bnn_mean + 1.96*y_pred_bnn_std, 
                     color='green', alpha=0.2, label='Incertidumbre Predictiva (95% CI)') # Graficamos la banda de incertidumbre con 95% de confianza
    
    
    plt.axvspan(4, 7, color='gray', alpha=0.1, label='Fallo del Sensor (Zona Ciega)') #Sombreamos la zona ciega del robot
    
    plt.title('Comparativa de Modelos frente a Vacíos Epistémicos', fontsize=14) # Título de la gráfica
    plt.xlabel('Tiempo / Posición (x)', fontsize=12) # Título del eje x
    plt.ylabel('Señal Cinemática (y)', fontsize=12) # Título del eje y
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Mostramos la leyenda
    plt.tight_layout() # Ajustamos el layout
    
    # Guardamos la imagen
    os.makedirs('reports/figures', exist_ok=True)
    filepath = 'reports/figures/comparativa_robot.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight') # Guardamos la gráfica
    print(f"Gráfica guardada en '{filepath}'")
    
    plt.show() # Mostramos la gráfica

def plot_uncertainty_decomposition(X_test, y_pred_bnn_std, sigma_aleatoria_std):
    """
    Dibuja la evolución espacial de las componentes de la incertidumbre.
    Aísla la magnitud de la duda del modelo frente a la posición.
    """
    print("\n--- Generando Gráfica de Descomposición de Incertidumbre ---")
    plt.figure(figsize=(10, 5)) # Ajustamos el tamaño de la gráfica
    
    # Cálculos de varianzas y desviaciones para TODOS los puntos de la gráfica
    var_total = y_pred_bnn_std**2 
    var_aleatoria = sigma_aleatoria_std**2
    var_epistemica = np.maximum(var_total - var_aleatoria, 0.0)
    
    std_epistemica = np.sqrt(var_epistemica) 
    std_total = y_pred_bnn_std 
    
    # Graficamos las tres magnitudes de incertidumbre
    plt.plot(X_test, std_total, 'g-', linewidth=2, label=r'Incertidumbre Total ($\sigma_{total}$)')
    plt.plot(X_test, std_epistemica, 'purple', linestyle='--', linewidth=2.5, label=r'Incertidumbre Epistémica ($\sigma_{epist}$)')
    plt.axhline(y=sigma_aleatoria_std, color='orange', linestyle='-.', linewidth=2, label=r'Incertidumbre Aleatoria ($\sigma_{aleat}$)')
    
    
    plt.axvspan(4, 7, color='gray', alpha=0.1, label='Fallo del Sensor (Zona Ciega)') # Dibujamos la zona ciega
    
    # Formato y estilo
    plt.title('Descomposición Espacial de la Incertidumbre Bayesiana', fontsize=14) # Título de la gráfica
    plt.xlabel('Tiempo / Posición (x)', fontsize=12) # Título del eje x
    plt.ylabel(r'Magnitud de la Incertidumbre ($\sigma$)', fontsize=12) # Título del eje y
    plt.legend(loc='upper left') # Mostramos la leyenda
    plt.grid(True, alpha=0.3) # Mostramos el grid
    plt.tight_layout() # Ajustamos el layout, importante para que no se corten las etiquetas
    
    # Guardado
    os.makedirs('reports/figures', exist_ok=True)
    filepath = 'reports/figures/descomposicion_incertidumbre.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight') # Guardamos la gráfica
    print(f"Gráfica guardada en '{filepath}'")
    
    plt.show()

def plot_mlp_decision_boundary(xx, yy, Z_mlp_entropy, X_train, y_train, X_ood):
    """
    Dibuja el mapa de calor de la Entropía Predictiva para el MLP con MC Dropout.
    """
    print("\n--- Generando Mapa de Fronteras de Decisión (MLP) ---")
    fig, ax = plt.subplots(figsize=(8, 6)) # Ajustamos el tamaño de la gráfica
    cmap_scatter = ListedColormap(['#2ca02c', '#1f77b4', '#ff7f0e']) # Colores para las etiquetas

    levels = np.linspace(0.0, 1.1, 20) # Fijamos la misma escala absoluta de entropía para ambos mapas.

    # Panel MLP con MC Dropout
    contour = ax.contourf(xx, yy, Z_mlp_entropy, alpha=0.7, cmap='magma_r', levels=levels) # Dibujamos el mapa de calor
    cbar = plt.colorbar(contour, ax=ax) # Dibujamos la barra de color
    cbar.set_label('Entropía Predictiva (Incertidumbre)', fontsize=11) # Etiqueta de la barra de color
    
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_scatter, edgecolors='k', s=30) # Dibujamos los puntos de entrenamiento
    ax.scatter(X_ood[:, 0], X_ood[:, 1], c='red', marker='X', s=80, edgecolors='k', label='Atacante (OOD)') # Dibujamos el atacante
    
    ax.set_title('A: MLP (MC Dropout) - Incertidumbre Implícita', fontsize=13, fontweight='bold') # Título de la gráfica
    ax.set_xlabel('Característica $X_1$') # Etiqueta del eje x
    ax.set_ylabel('Característica $X_2$') # Etiqueta del eje y
    ax.legend(loc='upper right') # Mostramos la leyenda

    plt.tight_layout()
    
    os.makedirs('reports/figures', exist_ok=True)
    filepath = 'reports/figures/fronteras_decision_mlp.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight') # Guardamos la gráfica
    print(f"Gráfica guardada en '{filepath}'")
    plt.show()

def plot_bnn_decision_boundary(xx, yy, Z_bnn_entropy, X_train, y_train, X_ood):
    """
    Dibuja el mapa de calor de laEntropía Predictiva para la BNN.
    """
    print("\n--- Generando Mapa de Fronteras de Decisión (BNN) ---")
    fig, ax = plt.subplots(figsize=(8, 6)) # Ajustamos el tamaño de la gráfica
    cmap_scatter = ListedColormap(['#2ca02c', '#1f77b4', '#ff7f0e']) # Colores para las etiquetas
    
    levels = np.linspace(0.0, 1.1, 20) # Fijamos la misma escala absoluta de entropía para ambos mapas.

    # Panel BNN
    contour = ax.contourf(xx, yy, Z_bnn_entropy, alpha=0.7, cmap='magma_r', levels=levels) # Dibujamos el mapa de calor
    cbar = plt.colorbar(contour, ax=ax) # Dibujamos la barra de color
    cbar.set_label('Entropía Predictiva (Incertidumbre)', fontsize=11) # Etiqueta de la barra de color
    
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_scatter, edgecolors='white', s=30) # Dibujamos los puntos de entrenamiento
    ax.scatter(X_ood[:, 0], X_ood[:, 1], c='red', marker='X', s=80, edgecolors='white', label='Atacante (OOD)') # Dibujamos el atacante
    
    ax.set_title('B: BNN (PyMC) - Incertidumbre Explícita', fontsize=13, fontweight='bold') # Título de la gráfica
    ax.set_xlabel('Característica $X_1$') # Titulo del eje X
    ax.set_ylabel('Característica $X_2$') # Títujlo del eje Y
    ax.legend(loc='upper right') # Mostramos la leyenda

    plt.tight_layout() # Ajustamos el layout, importante para que no se corten las etiquetas
    
    # Guardado automático
    os.makedirs('reports/figures', exist_ok=True)
    filepath = 'reports/figures/fronteras_decision_bnn.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight') # Guardamos la gráfica
    print(f"Gráfica guardada en '{filepath}'")
    plt.show()

def plot_roc_ood_detection(entropy_mlp_id, entropy_mlp_ood, entropy_bnn_id, entropy_bnn_ood):
    """
    Dibuja la Curva ROC evaluando la capacidad de los modelos para detectar 
    ataques (Out-of-Distribution) basándose en su Entropía Predictiva.
    """
    print("\n--- Generando Curva ROC de Detección de Ataques ---")
     
    y_true = np.concatenate([np.zeros(len(entropy_mlp_id)), np.ones(len(entropy_mlp_ood))]) # Preparamos las etiquetas verdaderas: 0 para datos conocidos , 1 para ataques
    
    # Concatenamos las entropías  
    scores_mlp = np.concatenate([entropy_mlp_id, entropy_mlp_ood]) 
    scores_bnn = np.concatenate([entropy_bnn_id, entropy_bnn_ood])
    
    # Calculamos ROC y AUC para MLP Clásico
    fpr_mlp, tpr_mlp, _ = roc_curve(y_true, scores_mlp)
    roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
    
    # Calculamos ROC y AUC para BNN
    fpr_bnn, tpr_bnn, _ = roc_curve(y_true, scores_bnn)
    roc_auc_bnn = auc(fpr_bnn, tpr_bnn)
    
    plt.figure(figsize=(8, 6)) # Ajustamos el tamaño de la gráfica
    
    plt.plot(fpr_mlp, tpr_mlp, color='#1f77b4', lw=2.5, linestyle='--', label=f'MLP Clásico (AUC = {roc_auc_mlp:.3f})') # Dibujamos la curva ROC del MLP 
    
    plt.plot(fpr_bnn, tpr_bnn, color='#2ca02c', lw=2.5, label=f'BNN Probabilística (AUC = {roc_auc_bnn:.3f})') # Dibujamos la curva ROC de la red bayesiana
    
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':', label='Azar (AUC = 0.500)') # Dibujamos la curva ROC del azar, para comparación
    
    plt.title('Curva ROC: Detección de Ataques (OOD) usando Entropía', fontsize=14, fontweight='bold') # Título de la gráfica
    plt.xlabel('Tasa de Falsos Positivos (Bloquear a usuarios legítimos)', fontsize=12) # Título del eje x
    plt.ylabel('Tasa de Verdaderos Positivos (Bloquear al atacante)', fontsize=12) # Título del eje y
    plt.legend(loc='lower right', fontsize=11) # Mostramos la leyenda
    plt.grid(True, alpha=0.3) # Mostramos el grid
    
    os.makedirs('reports/figures', exist_ok=True)
    filepath = 'reports/figures/roc_ood_detection.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight') # Guardamos la gráfica
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
    
    bins = np.linspace(0.33, 1.0, 30) # Intervalos para el histograma, de 0.33 a 1.0 con 30 intervalos
    
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
    plt.savefig(filepath, dpi=300, bbox_inches='tight') # Guardamos la gráfica
    print(f"Histograma de Confianzas guardado en '{filepath}'")
    
    plt.show()

def plot_elbo(mean_field):
    print("\n--- Generando Gráfica de Convergencia del Optimizador Variacional (Fase 2) ---") 
    plt.figure(figsize=(10, 4)) # Ajustamos el tamaño de la gráfica
    plt.plot(mean_field.hist, color='purple', label='Evolución Pérdida ELBO') # Dibujamos la evolución del ELBO
    plt.title('Convergencia del Optimizador Variacional (Fase 2 - hour.csv)') # Título de la gráfica
    plt.xlabel('Iteraciones') # Título del eje x
    plt.ylabel('Pérdida (ELBO)') # Título del eje y
    plt.legend() # Mostramos la leyenda
    plt.tight_layout() # Ajustamos el layout, importante para que no se corten las etiquetas
    
    os.makedirs('reports/figures', exist_ok=True) 
    filepath = 'reports/figures/elbo.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight') # Guardamos la gráfica
    print(f"Gráfica de Convergencia del Optimizador Variacional guardada en '{filepath}'")
    
    plt.show() # Mostramos la gráfica

def plot_bikes_mcmc_vs_advi(y_true, y_mean_mcmc, y_std_mcmc, y_mean_advi, y_std_advi, num_points=100):
    """
    Genera y guarda una gráfica comparativa con dos paneles para contrastar 
    la nitidez predictiva de MCMC frente a la sobre-cobertura de ADVI en el dataset de bicicletas.
    """
    print("\n--- Generando Gráfica Comparativa: MCMC vs ADVI (Fase 1) ---")
    
    # Seleccionamos un segmento representativo para que los puntos sean distinguibles.
    # Por defecto, los primeros 100 días del conjunto de prueba.
    n = min(num_points, len(y_true))
    x_axis = np.arange(n)
    
    # Creamos la figura con dos subplots compartiendo el eje X y el eje Y
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    
    # Panel A: MCMC
    axes[0].scatter(x_axis, y_true[:n], color='black', alpha=0.7, s=20, label='Demanda Real (Ground Truth)') # Dibujamos la función verdadera
    axes[0].plot(x_axis, y_mean_mcmc[:n], color='#1f77b4', linewidth=2, label='Media Predictiva MCMC') # Dibujamos la media predictiva
    axes[0].fill_between(x_axis, 
                         np.maximum(0, y_mean_mcmc[:n] - 1.96 * y_std_mcmc[:n]), 
                         y_mean_mcmc[:n] + 1.96 * y_std_mcmc[:n], 
                         color='#1f77b4', alpha=0.2, label='Incertidumbre 95% (Bien calibrada)') # Dibujamos la banda de incertidumbre
    
    axes[0].set_title('A: Inferencia Exacta (MCMC / NUTS) - Alta Nitidez Predictiva', fontsize=13, fontweight='bold') # Título del panel A
    axes[0].set_ylabel('Bicicletas Alquiladas', fontsize=11) # Etiqueta del eje Y
    axes[0].legend(loc='upper right') # Mostramos la leyenda
    axes[0].grid(True, alpha=0.3) # Mostramos el grid

    # Panel B: ADVI
    axes[1].scatter(x_axis, y_true[:n], color='black', alpha=0.7, s=20, label='Demanda Real (Ground Truth)') # Dibujamos la función verdadera
    axes[1].plot(x_axis, y_mean_advi[:n], color='#d62728', linewidth=2, label='Media Predictiva ADVI') # Dibujamos la media predictiva
    axes[1].fill_between(x_axis, 
                         np.maximum(0, y_mean_advi[:n] - 1.96 * y_std_advi[:n]), 
                         y_mean_advi[:n] + 1.96 * y_std_advi[:n], 
                         color='#d62728', alpha=0.2, label='Incertidumbre 95% (Sobre-cobertura por ruido)') # Dibujamos la banda de incertidumbre
    
    axes[1].set_title('B: Inferencia Variacional (ADVI) - Colapso por Aproximación de Campo Medio', fontsize=13, fontweight='bold') # Título del panel B
    axes[1].set_xlabel('Observaciones Temporales (Días de Test)', fontsize=11) # Etiqueta del eje X
    axes[1].set_ylabel('Bicicletas Alquiladas', fontsize=11)# Etiqueta del eje Y
    axes[1].legend(loc='upper right') # Mostramos la leyenda
    axes[1].grid(True, alpha=0.3) # Mostramos el grid

    plt.tight_layout() # Ajustamos el layout
    
    os.makedirs('reports/figures', exist_ok=True)
    filepath = 'reports/figures/comparativa_bikes_fase1.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight') # Guardamos la gráfica
    print(f"Gráfica guardada en '{filepath}'")
    
    plt.show()