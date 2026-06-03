import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BaselineClassifier(nn.Module):
    """
    Red Neuronal Determinista Mejorada.
    Incluye Dropout para regularización espacial de las fronteras. 
    La arquitectura consiste en dos capas ocultas de 32 neuronas cada una con función de activación ReLU, seguida de una capa de salida para clasificación multiclase.
    """
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=3):
        # Inicialización de la red neuronal
        super(BaselineClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim) #Capa oculta
        self.relu = nn.ReLU() #Función de activación ReLU
        # Dropout: apaga el 30% de las neuronas aleatoriamente en cada paso de entrenamiento
        self.dropout = nn.Dropout(p=0.3)  
        self.layer2 = nn.Linear(hidden_dim, output_dim) #Capa de salida para clasificación multiclase
        
    def forward(self, x):
        out = self.layer1(x) #Capa oculta
        out = self.relu(out) #Función de activación ReLU
        out = self.dropout(out) # Aplicamos Dropout después de la activación para regularizar la red y evitar sobreajuste, especialmente en regiones de frontera complejas
        out = self.layer2(out) #Capa de salida para clasificación multiclase
        return out

def train_classifier(model, X_train, y_train, num_epochs=600, learning_rate=0.01):
    """
    Bucle de entrenamiento clásico usando entropía cruzada y L2.
    """
    print("Iniciando entrenamiento del MLP (Mejorado con Dropout y L2)...")
    model.train() # Importante: activar el modo entrenamiento para que el Dropout funcione
    loss_function = nn.CrossEntropyLoss()
    
    # Regularización L2 para penalizar pesos extremadamente grandes
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    X_tensor = torch.FloatTensor(X_train) # Convertir datos de entrenamiento a tensor
    y_tensor = torch.LongTensor(y_train) # Convertir etiquetas de entrenamiento a tensor (LongTensor para clasificación)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad() # Reiniciar los gradientes para evitar acumulación
        outputs = model(X_tensor) # Forward pass
        loss = loss_function(outputs, y_tensor) # Cálculo de la pérdida usando entropía cruzada
        loss.backward() # Retropagación hacia atrás para calcular los gradientes
        optimizer.step() # Actualización de los pesos por descenso de gradiente
        
        if (epoch + 1) % 150 == 0:
            print(f" Época [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            
    return model

def predict_classifier(model, X_test, temperature=2.5):
    """
    Genera las predicciones de probabilidad utilizando Temperature Scaling.
    """
    model.eval() # Importante: desactivar Dropout para la inferencia
    X_tensor = torch.FloatTensor(X_test) # Convertir datos de prueba a tensor
    with torch.no_grad():
        logits = model(X_tensor) # Salidas sin normalizar (logits) del modelo
        
        # Aplicamos Temperature Scaling para suavizar la sobreconfianza
        scaled_logits = logits / temperature # Dividir los logits por la temperatura para suavizar las probabilidades, especialmente en regiones de frontera donde el modelo puede ser demasiado confiado
        
        probs = torch.softmax(scaled_logits, dim=1).numpy() # Convertir a probabilidades usando softmax y luego a numpy array para facilitar su uso posterior
    return probs

def predict_mc_dropout(model, X_test, num_samples=50):
    """
    Inferencia estocástica usando Monte Carlo Dropout.
    Convierte el MLP en un estimador de incertidumbre epistémica.
    """
    model.train() # Mantenemos el Dropout ACTIVADO durante la inferencia
    X_tensor = torch.FloatTensor(X_test)
    
    probs_samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(X_tensor) # Salidas sin normalizar (logits) del modelo con Dropout activo, lo que introduce variabilidad en las predicciones y permite estimar la incertidumbre epistémica, especialmente útil en regiones de frontera donde el modelo puede ser menos confiado
            probs = torch.softmax(logits, dim=1).numpy() # Convertir a probabilidades usando softmax y luego a numpy array para facilitar su uso posterior
            probs_samples.append(probs) # Agregar las probabilidades a la lista
            
    # Convertimos a array: shape (num_samples, n_test_samples, n_classes)
    probs_samples = np.array(probs_samples)
    
    # La probabilidad predictiva es la media geométrica de las pasadas estocásticas
    p_mean = probs_samples.mean(axis=0)
    
    return probs_samples,p_mean
