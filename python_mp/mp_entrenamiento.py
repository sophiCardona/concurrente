import numpy as np
import time
import multiprocessing
import os
import sys

# Añadir el directorio python_secuencial al path para poder importar
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_secuencial'))
from preprocesamiento import obtener_datos_listos

# --- 1. FUNCIONES MATEMÁTICAS (Globales para que los workers las vean) ---

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def relu_derivative(Z):
    return Z > 0

# --- 2. TAREA DEL WORKER (Calcula Gradientes, NO actualiza) ---
def worker_compute_gradients(W1, b1, W2, b2, X_chunk, y_chunk):
    """
    Esta función se ejecuta en un núcleo separado.
    Recibe una copia de los pesos y un pedazo de los datos.
    Devuelve los gradientes calculados para ese pedazo.
    """
    m = X_chunk.shape[0]
    
    # A. Forward Propagation
    Z1 = np.dot(X_chunk, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    
    # B. Backward Propagation (Calcular Gradientes)
    
    # Error en salida
    dZ2 = A2 - y_chunk
    
    # Gradientes Capa 2
    dW2 = np.dot(A1.T, dZ2) # Nota: No dividimos por 'm' aquí, lo haremos al promediar en el maestro
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # Error en Oculta
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    
    # Gradientes Capa 1
    dW1 = np.dot(X_chunk.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2, m

# --- 3. CLASE MAESTRA (Orquestador) ---
class ParallelMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialización de pesos (Igual que antes)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward_predict(self, X):
        # Solo para medir accuracy (secuencial)
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        return softmax(Z2)

    def update_weights(self, total_gradients, learning_rate, total_samples):
        # Desempaquetar la suma de gradientes
        sum_dW1, sum_db1, sum_dW2, sum_db2 = total_gradients
        
        # Promediar (dividir por el tamaño total del batch)
        # Gradient Descent: W = W - alpha * (grad_promedio)
        self.W1 -= learning_rate * (sum_dW1 / total_samples)
        self.b1 -= learning_rate * (sum_db1 / total_samples)
        self.W2 -= learning_rate * (sum_dW2 / total_samples)
        self.b2 -= learning_rate * (sum_db2 / total_samples)

# --- 4. UTILERÍAS ---
def calcular_precision(red, x, y):
    preds = red.forward_predict(x)
    return np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))

# --- 5. BLOQUE PRINCIPAL (Requerido para Multiprocessing en Windows) ---
if __name__ == "__main__":
    # Configuración
    NUM_WORKERS = 4 
    
    print(f"🚀 Iniciando Entrenamiento Paralelo con {NUM_WORKERS} procesos (Workers)...")
    
    # 1. Cargar Datos
    x_train, y_train, x_test, y_test = obtener_datos_listos()
    
    # 2. Hiperparámetros
    INPUT = 784
    HIDDEN = 512
    OUTPUT = 10
    LR = 0.1
    EPOCHS = 10         # 10 épocas para probar rápido
    BATCH_SIZE = 64 # El batch se dividirá entre los workers
    
    mlp = ParallelMLP(INPUT, HIDDEN, OUTPUT)
    
    # Crear Pool de Procesos
    # Usamos 'spawn' o el default. En Windows es obligatorio el bloque if __name__
    pool = multiprocessing.Pool(processes=NUM_WORKERS)
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Shuffle
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]
        
        # Loop por Batches
        for i in range(0, x_train.shape[0], BATCH_SIZE):
            x_batch = x_shuffled[i:i+BATCH_SIZE]
            y_batch = y_shuffled[i:i+BATCH_SIZE]
            
            # --- FASE PARALELA ---
            # 1. Dividir el batch en trozos para cada worker
            chunks_x = np.array_split(x_batch, NUM_WORKERS)
            chunks_y = np.array_split(y_batch, NUM_WORKERS)
            
            # 2. Preparar argumentos para cada worker
            # (Pesos actuales, trozo de X, trozo de Y)
            tasks = []
            for j in range(NUM_WORKERS):
                # OJO: Si el trozo está vacío (batch final pequeño), ignorar
                if len(chunks_x[j]) > 0:
                    args = (mlp.W1, mlp.b1, mlp.W2, mlp.b2, chunks_x[j], chunks_y[j])
                    tasks.append(args)
            
            # 3. Enviar a los workers y esperar resultados
            # starmap desempaqueta los argumentos automáticamente
            results = pool.starmap(worker_compute_gradients, tasks)
            
            # 4. Recolectar (Reduce)
            total_dW1 = np.zeros_like(mlp.W1)
            total_db1 = np.zeros_like(mlp.b1)
            total_dW2 = np.zeros_like(mlp.W2)
            total_db2 = np.zeros_like(mlp.b2)
            total_samples = 0
            
            for res in results:
                dW1, db1, dW2, db2, n_samples = res
                total_dW1 += dW1
                total_db1 += db1
                total_dW2 += dW2
                total_db2 += db2
                total_samples += n_samples
            
            # 5. Actualizar Pesos (Maestro)
            if total_samples > 0:
                mlp.update_weights((total_dW1, total_db1, total_dW2, total_db2), LR, total_samples)
        
        # Evaluar
        acc = calcular_precision(mlp, x_train, y_train)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Accuracy: {acc*100:.2f}%")

    end_time = time.time()
    pool.close()
    pool.join()
    
    print(f"\n⏱️ Tiempo Total (Multiprocessing): {end_time - start_time:.2f} s")
    print(f"Número de Workers: {NUM_WORKERS}")

    # Verificar en Test
    test_acc = calcular_precision(mlp, x_test, y_test)
    print(f"🏆 Test Accuracy: {test_acc*100:.2f}%")