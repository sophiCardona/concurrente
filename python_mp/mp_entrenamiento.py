"""
Entrenamiento Paralelo de MLP con Multiprocessing
Distribuye el cálculo de gradientes entre múltiples procesos CPU
"""

import numpy as np
import time
import multiprocessing
import os
import sys

# Importar desde python_secuencial
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_secuencial'))
from preprocesamiento import obtener_datos_listos

# ============================================================================
# FUNCIONES MATEMÁTICAS (Globales para workers)
# ============================================================================

def relu(Z):
    """ReLU: f(x) = max(0, x)"""
    return np.maximum(0, Z)

def softmax(Z):
    """Softmax con estabilidad numérica"""
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def relu_derivative(Z):
    """Derivada de ReLU"""
    return Z > 0

# ============================================================================
# FUNCIÓN WORKER (Ejecutada en procesos separados)
# ============================================================================

def worker_compute_gradients(W1, b1, W2, b2, X_chunk, y_chunk):
    """
    Calcula gradientes para un fragmento del batch en un proceso separado.
    
    Estrategia de paralelización:
    - Cada worker recibe una copia de los pesos actuales
    - Procesa su fragmento de datos independientemente
    - Retorna solo los gradientes calculados (no actualiza pesos)
    
    Args:
        W1, b1, W2, b2: Pesos y biases actuales de la red
        X_chunk: Fragmento de datos de entrada
        y_chunk: Fragmento de etiquetas
        
    Returns:
        tuple: (dW1, db1, dW2, db2, num_samples)
    """
    m = X_chunk.shape[0]
    
    # Forward Propagation
    Z1 = np.dot(X_chunk, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    
    # Backward Propagation (calcular gradientes)
    
    # Gradientes de capa de salida
    dZ2 = A2 - y_chunk
    dW2 = np.dot(A1.T, dZ2)  # Sin dividir por m (se promedia después)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # Gradientes de capa oculta
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X_chunk.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2, m

# ============================================================================
# CLASE MLP PARALELA (Proceso Maestro)
# ============================================================================

class ParallelMLP:
    """
    MLP con entrenamiento paralelo mediante multiprocessing.
    
    El proceso maestro:
    - Mantiene los pesos centralizados
    - Distribuye fragmentos del batch a workers
    - Recolecta y promedia gradientes
    - Actualiza los pesos
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """Inicializa pesos de la red"""
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward_predict(self, X):
        """Forward pass secuencial (solo para evaluación)"""
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        return softmax(Z2)

    def update_weights(self, total_gradients, learning_rate, total_samples):
        """
        Actualiza pesos usando gradientes promediados.
        
        Args:
            total_gradients: Suma de gradientes de todos los workers
            learning_rate: Tasa de aprendizaje
            total_samples: Número total de muestras procesadas
        """
        sum_dW1, sum_db1, sum_dW2, sum_db2 = total_gradients
        
        # Promediar gradientes y aplicar gradient descent
        self.W1 -= learning_rate * (sum_dW1 / total_samples)
        self.b1 -= learning_rate * (sum_db1 / total_samples)
        self.W2 -= learning_rate * (sum_dW2 / total_samples)
        self.b2 -= learning_rate * (sum_db2 / total_samples)

# ============================================================================
# UTILIDADES
# ============================================================================

def calcular_precision(red, x, y):
    """Calcula accuracy de la red"""
    preds = red.forward_predict(x)
    return np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))

# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Configuración de paralelización
    NUM_WORKERS = 4
    
    print(f"=== Entrenamiento Paralelo MLP ===")
    print(f"Número de workers: {NUM_WORKERS}\n")
    
    # ---- Cargar datos ----
    x_train, y_train, x_test, y_test = obtener_datos_listos()
    
    # ---- Hiperparámetros ----
    INPUT = 784
    HIDDEN = 256
    OUTPUT = 10
    LR = 0.1
    EPOCHS = 10
    BATCH_SIZE = 64
    
    mlp = ParallelMLP(INPUT, HIDDEN, OUTPUT)
    
    # Crear pool de procesos
    pool = multiprocessing.Pool(processes=NUM_WORKERS)
    
    print(f"Hiperparámetros:")
    print(f"   Épocas: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LR}\n")
    
    start_time = time.time()
    
    # ---- Bucle de entrenamiento ----
    for epoch in range(EPOCHS):
        # Barajar datos
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]
        
        # Procesar por batches
        for i in range(0, x_train.shape[0], BATCH_SIZE):
            x_batch = x_shuffled[i:i+BATCH_SIZE]
            y_batch = y_shuffled[i:i+BATCH_SIZE]
            
            # ---- Fase paralela: Dividir batch entre workers ----
            
            # Dividir batch en fragmentos
            chunks_x = np.array_split(x_batch, NUM_WORKERS)
            chunks_y = np.array_split(y_batch, NUM_WORKERS)
            
            # Preparar tareas para cada worker
            tasks = []
            for j in range(NUM_WORKERS):
                if len(chunks_x[j]) > 0:  # Ignorar fragmentos vacíos
                    args = (mlp.W1, mlp.b1, mlp.W2, mlp.b2, chunks_x[j], chunks_y[j])
                    tasks.append(args)
            
            # Enviar tareas a workers y esperar resultados
            results = pool.starmap(worker_compute_gradients, tasks)
            
            # ---- Fase de reducción: Sumar gradientes ----
            
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
            
            # Actualizar pesos (proceso maestro)
            if total_samples > 0:
                mlp.update_weights((total_dW1, total_db1, total_dW2, total_db2), 
                                 LR, total_samples)
        
        # Evaluar accuracy al final de cada época
        acc = calcular_precision(mlp, x_train, y_train)
        print(f"Época {epoch+1}/{EPOCHS} | Train Accuracy: {acc*100:.2f}%")

    end_time = time.time()
    
    # Cerrar pool de procesos
    pool.close()
    pool.join()
    
    print(f"\n⏱️ Tiempo Total: {end_time - start_time:.2f} segundos")
    print(f"⏱️ Tiempo por época: {(end_time - start_time)/EPOCHS:.2f} segundos")

    # Evaluación en test set
    test_acc = calcular_precision(mlp, x_test, y_test)
    print(f"\n🏆 Test Accuracy: {test_acc*100:.2f}%")