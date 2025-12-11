"""
Preprocesamiento de Datos MNIST
Normaliza píxeles y convierte etiquetas a formato one-hot encoding
"""

import numpy as np
import os
from verificar_mnist import cargar_imagenes, cargar_etiquetas

# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================================

def one_hot_encode(etiquetas, num_clases=10):
    """
    Convierte etiquetas numéricas a formato one-hot encoding.
    
    Ejemplo: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    
    Args:
        etiquetas: Array de etiquetas numéricas (0-9)
        num_clases: Número total de clases
        
    Returns:
        numpy.ndarray: Matriz de forma (N, num_clases)
    """
    n = etiquetas.shape[0]
    one_hot = np.zeros((n, num_clases))
    one_hot[np.arange(n), etiquetas] = 1
    return one_hot

def preprocesar_datos(x, y):
    """
    Preprocesa imágenes y etiquetas para entrenamiento.
    
    Operaciones:
    1. Aplanar imágenes: (N, 28, 28) -> (N, 784)
    2. Normalizar píxeles: [0, 255] -> [0.0, 1.0]
    3. Codificar etiquetas a one-hot
    
    Args:
        x: Imágenes de forma (N, 28, 28)
        y: Etiquetas de forma (N,)
        
    Returns:
        tuple: (x_normalizado, y_one_hot)
    """
    num_imagenes = x.shape[0]
    
    # Aplanar y normalizar
    x_plano = x.reshape(num_imagenes, 784)
    x_norm = x_plano / 255.0
    
    # One-hot encoding
    y_encoded = one_hot_encode(y)
    
    return x_norm, y_encoded

# ============================================================================
# FUNCIÓN DE CARGA COMPLETA
# ============================================================================

def obtener_datos_listos():
    """
    Función principal: Carga y preprocesa todos los datos MNIST.
    
    Realiza:
    1. Carga archivos .gz desde ../data/
    2. Normaliza y aplana imágenes
    3. Convierte etiquetas a one-hot
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test)
            - x_train: (60000, 784) valores en [0.0, 1.0]
            - y_train: (60000, 10) one-hot
            - x_test: (10000, 784) valores en [0.0, 1.0]  
            - y_test: (10000, 10) one-hot
    """
    print("--- Cargando datos MNIST ---")
    
    # Obtener ruta del directorio de datos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    
    # Cargar archivos
    x_train = cargar_imagenes(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    y_train = cargar_etiquetas(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    x_test = cargar_imagenes(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = cargar_etiquetas(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    print("--- Preprocesando datos ---")
    x_train_final, y_train_final = preprocesar_datos(x_train, y_train)
    x_test_final, y_test_final = preprocesar_datos(x_test, y_test)
    
    print("✅ Datos listos para entrenamiento")
    return x_train_final, y_train_final, x_test_final, y_test_final

# ============================================================================
# PROGRAMA DE PRUEBA
# ============================================================================

if __name__ == "__main__":
    x_tr, y_tr, x_te, y_te = obtener_datos_listos()

    print(f"\nForma de datos finales:")
    print(f"  X_train: {x_tr.shape}")
    print(f"  Y_train: {y_tr.shape}")
    print(f"  X_test:  {x_te.shape}")
    print(f"  Y_test:  {y_te.shape}")
    print(f"\nEjemplo de one-hot: {y_tr[0]}")
    print("¡Todo listo para entrenamiento!")