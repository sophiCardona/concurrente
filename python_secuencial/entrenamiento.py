"""
Entrenamiento de Red Neuronal MLP - Versión Secuencial
Implementa backpropagation desde cero con NumPy
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# FUNCIONES DE ACTIVACIÓN
# ============================================================================

def relu(Z):
    """Rectified Linear Unit: f(x) = max(0, x)"""
    return np.maximum(0, Z)

def relu_derivative(Z):
    """
    Derivada de ReLU:
    f'(x) = 1 si x > 0, 0 en caso contrario
    """
    return Z > 0

def softmax(Z):
    """
    Softmax: Convierte logits en distribución de probabilidad.
    
    Incluye estabilidad numérica restando el máximo de cada fila.
    """
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# ============================================================================
# FUNCIÓN DE PÉRDIDA
# ============================================================================

def cross_entropy_loss(y_pred, y_real):
    """
    Categorical Cross-Entropy Loss.
    
    Mide la diferencia entre predicciones y etiquetas reales.
    Añade epsilon para evitar log(0).
    
    Args:
        y_pred: Probabilidades predichas (N, 10)
        y_real: Etiquetas one-hot (N, 10)
        
    Returns:
        float: Pérdida promedio
    """
    m = y_real.shape[0]
    loss = -np.sum(y_real * np.log(y_pred + 1e-9)) / m
    return loss

# ============================================================================
# CLASE MLP - RED NEURONAL
# ============================================================================

class MLP:
    """
    Multi-Layer Perceptron de 3 capas.
    
    Arquitectura:
        Entrada (784) -> Oculta (hidden_size, ReLU) -> Salida (10, Softmax)
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Inicializa pesos con valores aleatorios pequeños.
        
        Args:
            input_size: Número de características de entrada (784)
            hidden_size: Número de neuronas en capa oculta
            output_size: Número de clases de salida (10)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicialización de pesos (Xavier simplificado)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        print(f"🧠 MLP Inicializado: {input_size} -> {hidden_size} (ReLU) -> {output_size} (Softmax)")

    def forward(self, X):
        """
        Forward Propagation: Calcula predicciones.
        
        Paso 1: Z1 = X·W1 + b1  →  A1 = ReLU(Z1)
        Paso 2: Z2 = A1·W2 + b2 →  A2 = Softmax(Z2)
        
        Args:
            X: Batch de entrada (batch_size, 784)
            
        Returns:
            numpy.ndarray: Probabilidades de salida (batch_size, 10)
        """
        # Capa oculta
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        
        # Capa de salida
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        
        return self.A2
    
    def backward(self, X, y, learning_rate=0.1):
        """
        Backward Propagation: Calcula gradientes y actualiza pesos.
        
        Algoritmo:
        1. Calcular error de salida: dZ2 = A2 - y
        2. Calcular gradientes de W2, b2
        3. Propagar error a capa oculta
        4. Calcular gradientes de W1, b1
        5. Actualizar todos los parámetros
        
        Args:
            X: Batch de entrada (batch_size, 784)
            y: Etiquetas one-hot (batch_size, 10)
            learning_rate: Tasa de aprendizaje
        """
        m = X.shape[0]  # Tamaño del batch
        
        # ---- Gradientes de capa de salida ----
        
        # Error: dZ2 = A2 - y (derivada de softmax + cross-entropy)
        dZ2 = self.A2 - y
        
        # Gradientes de W2 y b2
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # ---- Gradientes de capa oculta ----
        
        # Propagar error hacia atrás
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        
        # Gradientes de W1 y b1
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # ---- Actualización de parámetros (Gradient Descent) ----
        
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

# ============================================================================
# UTILIDADES
# ============================================================================

def calcular_precision(red, x, y_one_hot):
    """
    Calcula accuracy de la red en un conjunto de datos.
    
    Args:
        red: Instancia de MLP
        x: Datos de entrada
        y_one_hot: Etiquetas en formato one-hot
        
    Returns:
        float: Accuracy (porcentaje de aciertos)
    """
    predicciones = red.forward(x)
    pred_labels = np.argmax(predicciones, axis=1)
    true_labels = np.argmax(y_one_hot, axis=1)
    aciertos = np.sum(pred_labels == true_labels)
    return aciertos / x.shape[0]

# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

# Importar función de carga de datos
try:
    from preprocesamiento import obtener_datos_listos
except ImportError:
    print("❌ Error: No se encuentra 'preprocesamiento.py'")
    print("Asegúrate de que ambos archivos estén en la misma carpeta.")
    exit()

if __name__ == "__main__":
    print("=== Entrenamiento Secuencial MLP ===\n")
    
    # ---- Cargar y preparar datos ----
    x_train, y_train, x_test, y_test = obtener_datos_listos()
    
    # ---- Configurar hiperparámetros ----
    INPUT_SIZE = 784
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 10
    LEARNING_RATE = 0.1
    EPOCHS = 10
    BATCH_SIZE = 64
    
    # ---- Inicializar red ----
    mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    print(f"\n🚀 Iniciando entrenamiento:")
    print(f"   Épocas: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}\n")
    
    tiempo_inicio = time.time()

    # ---- Bucle de entrenamiento ----
    for epoch in range(EPOCHS):
        
        # Barajar datos cada época
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]
        
        # Procesar por mini-batches
        for i in range(0, x_train.shape[0], BATCH_SIZE):
            x_batch = x_shuffled[i:i+BATCH_SIZE]
            y_batch = y_shuffled[i:i+BATCH_SIZE]
            
            # Forward + Backward
            mlp.forward(x_batch)
            mlp.backward(x_batch, y_batch, learning_rate=LEARNING_RATE)
            
        # Evaluar al final de cada época
        acc = calcular_precision(mlp, x_train, y_train)
        prediccion_actual = mlp.forward(x_train)
        loss = cross_entropy_loss(prediccion_actual, y_train)
        
        print(f"Época {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

    tiempo_fin = time.time()
    tiempo_total = tiempo_fin - tiempo_inicio

    print(f"\n⏱ Tiempo total: {tiempo_total:.2f} segundos")
    print(f"⏱ Tiempo promedio por época: {tiempo_total/EPOCHS:.2f} segundos")
    
    # ---- Evaluación final en test set ----
    acc_test = calcular_precision(mlp, x_test, y_test)
    print(f"\n🏆 Accuracy en Test Set: {acc_test*100:.2f}%")

    # ---- Visualización de predicciones ----
    def probar_imagen_al_azar():
        """Muestra una predicción aleatoria con su distribución de probabilidades."""
        idx = np.random.randint(0, x_test.shape[0])
        
        img_input = x_test[idx:idx+1]
        etiqueta_real = np.argmax(y_test[idx])
        
        prediccion_probs = mlp.forward(img_input)
        prediccion_final = np.argmax(prediccion_probs)
        
        # Crear figura con 2 paneles
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Panel 1: Imagen
        imagen_2d = img_input.reshape(28, 28)
        ax1.imshow(imagen_2d, cmap='gray')
        ax1.set_title(f"Real: {etiqueta_real} | Predicción: {prediccion_final}")
        ax1.axis('off')
        
        # Panel 2: Distribución de probabilidades
        barras = ax2.bar(range(10), prediccion_probs[0])
        ax2.set_xticks(range(10))
        ax2.set_title("Probabilidad asignada a cada dígito")
        ax2.set_ylim(0, 1.1)
        ax2.set_xlabel("Dígito")
        ax2.set_ylabel("Probabilidad")
        
        # Colorear barra predicha (verde si correcto, rojo si incorrecto)
        barras[prediccion_final].set_color('green' if prediccion_final == etiqueta_real else 'red')
        
        plt.tight_layout()
        plt.show()

    # Mostrar 3 ejemplos de predicciones
    print("\nMostrando ejemplos de predicciones...")
    for _ in range(3):
        probar_imagen_al_azar()