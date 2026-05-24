import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class BaselineNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        #Inicialización de la red neuronal
        super(BaselineNN, self).__init__()
        #Definición de las capas de la red neuronal
        self.fc1 = nn.Linear(input_dim, hidden_dim) #Capa oculta
        self.relu = nn.ReLU() #Función de activación ReLU
        self.fc2 = nn.Linear(hidden_dim, output_dim) #Capa de salida

    def forward(self, x):
        #Propagación hacia adelante
        out = self.fc1(x) #Capa oculta
        out = self.relu1(out) #Función de activación ReLU
        out = self.fc2(out) #Capa de salida
        out = self.relu2(out) #Función de activación ReLU
        out = self.fc3(out) #Capa de salida
        return out

def train_model(model, X_train, y_train, num_epochs=1000, learning_rate=0.001):
    #Función de pérdida y optimizador
    criterion = nn.MSELoss() #Función de pérdida usando error cuadrático medio
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Optimizador Adam: actualiza los pesos de la red neuronal
    
    X_train = torch.tensor(X_train, dtype=torch.float32) #Convertir datos de entrenamiento a tensor
    y_train = torch.tensor(y_train, dtype=torch.float32) #Convertir etiquetas de entrenamiento a tensor

    #Entrenamiento del modelo
    for epoch in range(num_epochs):

        model.train() #Modo entrenamiento
        predictions = model(X_train) #Predicciones del modelo
        loss = criterion(predictions, y_train) #Cálculo de la pérdida
        optimizer.zero_grad() #Reiniciar los gradientes
        loss.backward() #Retropagación hacia atrás
        optimizer.step() #Actualización de los pesos por descenso de gradiente

        if (epoch+1) % 200 == 0:
            print(f'Época [{epoch+1}/{num_epochs}], Pérdida(Error Medio Cuadrático): {loss.item():.4f}')
    return model
def predict_model(model, X_test):
    # Generar predicciones puntuales para un nuevo conjunto de datos de prueba
    model.eval() #Modo de evaluación, desactiva el dropout y la normalización por lotes
    with torch.no_grad(): #Desactiva el cálculo de gradientes
        X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) #Convertir datos de prueba a tensor, usamos unsqueeze para agregar una dimensión adicional para que sea compatible con la entrada de la red neuronal
        predictions = model(X_test) #Generar predicciones
    return predictions.squeeze().numpy() #Devolver las predicciones como un array de NumPy, usamos squeeze para eliminar la dimensión adicional que agregamos anteriormente