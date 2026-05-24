import numpy as np
def rmse(y_true, y_pred): 
    """
    Calcula el Error Cuadrático Medio Raíz (RMSE).
    Mide la precisión puntual de la predicción.
    """
    return np.sqrt(np.mean((y_true - y_pred)**2)) #Devuelve la raíz cuadrada del error cuadrático medio, lo que proporciona una medida de la precisión de las predicciones del modelo en las mismas unidades que los datos originales, facilitando la interpretación de los resultados.


def decompose_uncertainty(var_total, sigma2_mean): 
    """
    Descompone la incertidumbre total en sus componentes aleatoria y epistémica.
    var_total: Varianza total de la distribución predictiva a posteriori.
    sigma2_mean: Media de la varianza aleatoria (sigma^2) de la distribución predictiva a posteriori.
    """
    epi = np.maximum(var_total - sigma2_mean, 0.0) 
    return sigma2_mean, epi #Devuelve la incertidumbre aleatoria (sigma2_mean) y la incertidumbre epistémica (epi) por separado, lo que permite un análisis más detallado de las fuentes de incertidumbre en las predicciones del modelo.

def calculate_coverage_95(y_true, y_pred_mean, y_pred_std):
    """
    Calcula la Probabilidad de Cobertura del Intervalo de Predicción (PICP) al 95%.
    Mide qué porcentaje de los datos reales cae dentro de nuestras bandas de incertidumbre.
    Idealmente, debería acercarse al 95%.
    """

    #El factor 1.96 se deriva de la distribución normal estándar y corresponde a un intervalo de confianza del 95%, lo que significa que aproximadamente el 95% de los datos reales deberían caer dentro de este rango si las predicciones del modelo son precisas y la incertidumbre está bien calibrada.
    lower_bound = y_pred_mean - 1.96 * y_pred_std 
    upper_bound = y_pred_mean + 1.96 * y_pred_std
    
    # Vector booleano: True si el dato real está en el intervalo, False si no
    in_bounds = (y_true >= lower_bound) & (y_true <= upper_bound)
    
    coverage = np.mean(in_bounds) * 100 # Calculamos el porcentaje de datos reales que caen dentro del intervalo de predicción, lo que nos da una medida de la cobertura del modelo. Un valor cercano al 95% indicaría que el modelo está bien calibrado en términos de su incertidumbre.
    return coverage

def calculate_mean_predictive_variance(y_pred_std):
    """
    Calcula la varianza predictiva media.
    Útil para cuantificar cómo "explota" la incertidumbre en zonas ciegas, ya que tanto el RMSE como el PICP pueden ser engañosos en estas situaciones 
    pues dependen de los valores reales y de las predicciones puntuales.
    """
    return np.mean(y_pred_std**2) #Devuelve la varianza predictiva media, que es una medida de la incertidumbre promedio en las predicciones del modelo. Un valor alto puede indicar que el modelo tiene dificultades para hacer predicciones precisas en ciertas áreas del espacio de entrada, lo que podría ser un indicio de zonas ciegas o falta de datos representativos en esas regiones.