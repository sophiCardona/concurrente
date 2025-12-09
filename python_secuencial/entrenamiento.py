import numpy as np
import matplotlib.pyplot as plt
import time 

# función para calcular la pérdida
def cross_entropy_loss(y_pred, y_real):
    """
    Calcula la pérdida Categorical Cross-Entropy.
    y_pred: Probabilidades predichas por la red (A2)
    y_real: Etiquetas en formato One-Hot
    """
    m = y_real.shape[0]
    # Se agrega un epsilon muy pequeño (1e-9) dentro del log para evitar log(0) que da error
    loss = -np.sum(y_real * np.log(y_pred + 1e-9)) / m
    return loss

# funcion para backpropagation 
def relu_derivative(Z):
    """
    Derivada de ReLU:
    Si Z > 0, la pendiente es 1.
    Si Z <= 0, la pendiente es 0.
    Devuelve True/False que funciona como 1/0 en matemáticas.
    """
    return Z > 0
# --- FUNCIONES DE ACTIVACIÓN (Matemáticas puras) ---
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    # Restamos el max para estabilidad numérica
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# --- CLASE PRINCIPAL ---
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Inicializa los pesos y sesgos de la red.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicialización de He/Xavier simplificada (Random normal * factor pequeño)
        # W1: Conecta Entrada (784) -> Oculta (N)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        
        # W2: Conecta Oculta (N) -> Salida (10)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        print(f"🧠 MLP Inicializado: {input_size} -> {hidden_size} (ReLU) -> {output_size} (Softmax)")

    def forward(self, X):
        """
        Realiza la predicción (Forward Propagation).
        Pasa los datos X a través de la red.
        """
        # 1. Capa Oculta
        # Z1 = X * W1 + b1
        self.Z1 = np.dot(X, self.W1) + self.b1
        # A1 = Función de activación (ReLU) aplicado a Z1
        self.A1 = relu(self.Z1)
        
        # 2. Capa de Salida
        # Z2 = A1 * W2 + b2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        # A2 = Probabilidades finales (Softmax)
        self.A2 = softmax(self.Z2)
        
        return self.A2
    
    def backward(self, X, y, learning_rate=0.1):
        """
        Backpropagation: Calcula culpables y actualiza pesos.
        X: Datos de entrada del lote actual
        y: Etiquetas reales (One-Hot)
        learning_rate: Qué tan rápido aprendemos (paso de actualización)
        """
        m = X.shape[0] # Tamaño del lote (batch size)
        
        # --- PARTE A: CALCULAR GRADIENTES (¿Quién tuvo la culpa del error?) ---
        
        # 1. Error en la Salida (Capa 2)
        # Matemáticamente, la derivada de Softmax + CrossEntropy es simple: (Predicción - Real)
        dZ2 = self.A2 - y
        
        # ¿Cuánto cambiar W2? (Entrada de la capa * Error)
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # 2. Error en la Oculta (Capa 1)
        # Llevamos el error hacia atrás: (Error Salida * Pesos W2) * Derivada ReLU
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        
        # ¿Cuánto cambiar W1?
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # --- PARTE B: ACTUALIZAR PESOS (Descenso del Gradiente) ---
        # Regla: Peso Nuevo = Peso Viejo - (Tasa * Gradiente)
        # Nos movemos en dirección opuesta al error.
        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1
        self.W2 = self.W2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2

#  importar la función maestra
try:
    from preprocesamiento import obtener_datos_listos

except ImportError:
    print("❌ Error: No encuentro el archivo 'preprocesamiento.py' o la función.")
    print("Asegúrate de que ambos archivos estén en la misma carpeta.")
    exit()

def probar_carga():
    print("Iniciando sistema...")
    
    # 1. Traer los datos
    x_train, y_train, x_test, y_test = obtener_datos_listos()
    
    print("\n Datos importados correctamente al script principal.")
    
    # 2. Verificaciones de Sanidad (Sanity Checks)
    # Confirmar dimensiones para la arquitectura MLP
    assert x_train.shape == (60000, 784), f"Error en dimensión X Train: {x_train.shape}"
    assert y_train.shape == (60000, 10),  f"Error en dimensión Y Train: {y_train.shape}"
    assert x_test.shape  == (10000, 784), f"Error en dimensión X Test: {x_test.shape}"
    assert y_test.shape  == (10000, 10),  f"Error en dimensión Y Test: {y_test.shape}"
    
    # Confirmar normalización (debe ser float entre 0 y 1)
    max_val = np.max(x_train)
    min_val = np.min(x_train)
    
    if max_val > 1.0 or min_val < 0.0:
        print(f"⚠️ ALERTA: Los datos no parecen normalizados. Max: {max_val}, Min: {min_val}")
    else:
        print(f"✅ Normalización correcta (Valores entre {min_val} y {max_val}).")

    print("\n--- Resumen para la Arquitectura MLP ---")
    print(f"Neuronas de Entrada requeridas: {x_train.shape[1]}") # 784
    print(f"Neuronas de Salida requeridas:  {y_train.shape[1]}") # 10
    print("Todo listo para construir la clase MLP.")

def calcular_precision(red, x, y_one_hot):
    # Función auxiliar para ver qué tan bien vamos
    predicciones = red.forward(x)
    pred_labels = np.argmax(predicciones, axis=1) # Elige la neurona con mayor valor
    true_labels = np.argmax(y_one_hot, axis=1)    # Elige el 1 en el one-hot
    aciertos = np.sum(pred_labels == true_labels)
    return aciertos / x.shape[0]

if __name__ == "__main__":
    # 1. Cargar
    x_train, y_train, x_test, y_test = obtener_datos_listos()
    
    # 2. Configurar Hiperparámetros
    INPUT_SIZE = 784
    HIDDEN_SIZE = 512   # Neuronas ocultas
    OUTPUT_SIZE = 10    # Clases (0-9)
    LEARNING_RATE = 0.1 # Velocidad de aprendizaje
    EPOCHS = 10         # Cuántas veces ver el dataset completo
    BATCH_SIZE = 64     # Procesar 64 imágenes a la vez (más estable)
    
    # 3. Iniciar Red
    mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    print(f"\n🚀 Iniciando entrenamiento: {EPOCHS} épocas, Batch {BATCH_SIZE}, LR {LEARNING_RATE}")
    
    tiempo_inicio = time.time()

    # --- CICLO DE ENTRENAMIENTO ---
    for epoch in range(EPOCHS):
        
        # A. Barajar datos (Shuffle)
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]
        
        # B. Procesar por lotes (Mini-batches)
        for i in range(0, x_train.shape[0], BATCH_SIZE):
            # Recortar el lote actual
            x_batch = x_shuffled[i:i+BATCH_SIZE]
            y_batch = y_shuffled[i:i+BATCH_SIZE]
            
            # MAGIA: Predecir y Ajustar
            mlp.forward(x_batch)
            mlp.backward(x_batch, y_batch, learning_rate=LEARNING_RATE)
            
        # C. Evaluar al final de la época
        # 1. Calcular Precisión
        acc = calcular_precision(mlp, x_train, y_train)
        
        # 2. Calcular PÉRDIDA (LOSS) - El requisito que faltaba
        # Necesitamos la predicción actual completa (probabilidades)
        prediccion_actual = mlp.forward(x_train)
        loss = cross_entropy_loss(prediccion_actual, y_train)
        
        print(f"Época {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

    tiempo_fin = time.time()
    tiempo_total = tiempo_fin - tiempo_inicio

    print(f"⏱ Tiempo total de entrenamiento: {tiempo_total:.2f} segundos")
    print(f" Tiempo promedio por época: {tiempo_total/EPOCHS:.2f} segundos")
    print("\n Entrenamiento finalizado.")
    
    # Evaluación final con datos que la red NUNCA ha visto (Test Set)
    acc_test = calcular_precision(mlp, x_test, y_test)
    print(f"🏆 Precisión Final en TEST SET: {acc_test*100:.2f}%")

    def probar_imagen_al_azar():
        # Elegir un índice aleatorio del set de prueba
        idx = np.random.randint(0, x_test.shape[0])
        
        # Tomar la imagen y su etiqueta real
        img_input = x_test[idx:idx+1]  # Mantenemos dimensión (1, 784)
        etiqueta_real = np.argmax(y_test[idx])
        
        # Preguntarle a la red
        prediccion_probs = mlp.forward(img_input)
        prediccion_final = np.argmax(prediccion_probs)
        
        # --- GRAFICAR ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # 1. La Imagen
        # Reconvertimos de 784 a 28x28 para verla
        imagen_2d = img_input.reshape(28, 28)
        ax1.imshow(imagen_2d, cmap='gray')
        ax1.set_title(f"Real: {etiqueta_real} | Predicción: {prediccion_final}")
        ax1.axis('off')
        
        # 2. Las Probabilidades (Lo que "piensa" la red)
        barras = ax2.bar(range(10), prediccion_probs[0])
        ax2.set_xticks(range(10))
        ax2.set_title("Probabilidad asignada a cada número")
        ax2.set_ylim(0, 1.1)
        
        # Colorear la barra elegida
        barras[prediccion_final].set_color('green' if prediccion_final == etiqueta_real else 'red')
        
        plt.show()

    # ¡Probemos con 3 ejemplos!
    print("\nMostrando ejemplos visuales...")
    for _ in range(3):
        probar_imagen_al_azar()