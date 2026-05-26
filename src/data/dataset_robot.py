import numpy as np

def generate_robot_data(n_samples=200, noise_std=0.5):
    """
    Genera la cinemática del robot: f(x) = 2*sin(1.5x) + 0.5*x
    Inyecta ruido aleatorio y crea un vacío epistémico (fallo de sensor) entre x=4 y x=7.
    """
    
    # Dominio completo

    # Generamos un conjunto de 500 puntos equiespaciados entre 0 y 10 para representar el dominio completo de la función, lo que nos permitirá 
    # evaluar el comportamiento del modelo en toda la gama de entradas, incluyendo las zonas con datos y las zonas ciegas.
    X_full = np.linspace(0, 10, 500) 

    # Calculamos los valores verdaderos de la función para el dominio completo, lo que nos servirá como referencia para evaluar la precisión 
    # de las predicciones del modelo y analizar cómo maneja la incertidumbre en diferentes regiones del espacio de entrada.
    y_true = 2 * np.sin(1.5 * X_full) + 0.5 * X_full 


    # Datos de entrenamiento (simulando el fallo del sensor)
    # Generamos un conjunto de datos de entrenamiento con n_samples puntos aleatorios distribuidos uniformemente entre 0 y 10, lo que nos permitirá
    # simular un escenario realista donde los datos no están ordenados ni agrupados.
    X_train = np.random.uniform(0, 10, n_samples) 

    # ELIMINAMOS los datos en el intervalo [4, 7] para simular un fallo del sensor, lo que crea una zona ciega en el espacio de entrada.
    mask = (X_train < 4) | (X_train > 7)
    X_train = X_train[mask] 
    
    # Añadimos ruido gaussiano (Incertidumbre aleatoria)
    # Generamos las etiquetas de entrenamiento añadiendo ruido gaussiano a los valores verdaderos de la función, lo que simula la incertidumbre 
    # aleatoria en las mediciones del sensor y permite evaluar cómo el modelo maneja esta incertidumbre en sus predicciones.
    y_train = 2 * np.sin(1.5 * X_train) + 0.5 * X_train + np.random.normal(0, noise_std, size=X_train.shape) 
    
    return X_train, y_train, X_full, y_true