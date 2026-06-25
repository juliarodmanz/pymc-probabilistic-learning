import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def cargar_datos_diarios(filepath='src\datasets\day.csv', test_size=0.2, random_state=2003):
    """
    Ingesta y estandarización del dataset agregado por días.
    Retorna tensores escalados y los objetos StandardScaler para la desnormalización posterior.
    """
    # 1. Carga del CSV (731 observaciones)
    df = pd.read_csv(filepath)
    
    # 2. Selección de características y variable objetivo 
    # En 'day.csv' no existe la variable 'hr' que se utiliza en 'hour.csv'
    features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 
                'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    
    X = df[features].values # Matriz de características
    y = df['cnt'].values # Vector de variable objetivo
    
    #  División del conjunto de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state) # Dividimos el conjunto de datos en conjuntos de entrenamiento y prueba
    
    # Estandarización Z-score (Crucial para la estabilidad de la matriz Jacobiana en MCMC)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train) # Estandarizamos el conjunto de entrenamiento
    X_test_scaled = scaler_X.transform(X_test) # Estandarizamos el conjunto de prueba
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten() # Estandarizamos el conjunto de entrenamiento
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten() # Estandarizamos el conjunto de prueba
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y


def cargar_datos_horarios(filepath='src\datasets\hour.csv', test_size=0.2, random_state=2003):
    """
    Ingesta y estandarización del dataset por horas.
    Retorna tensores escalados y los objetos StandardScaler.
    """
    # Carga del CSV (17.379 observaciones)
    df = pd.read_csv(filepath)
    
    #  Selección de características y variable objetivo
    # Se incluye la variable geométrica temporal 'hr'
    features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 
                'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    
    X = df[features].values # Matriz de características
    y = df['cnt'].values # Vector de variable objetivo
    
    # División del conjunto de datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Estandarización Z-score (Fundamental para la convergencia del gradiente estocástico)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train) # Estandarizamos el conjunto de entrenamiento
    X_test_scaled = scaler_X.transform(X_test) # Estandarizamos el conjunto de prueba
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten() # Estandarizamos el conjunto de entrenamiento
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten() # Estandarizamos el conjunto de prueba
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y