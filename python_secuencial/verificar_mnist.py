"""
Verificación y Carga del Dataset MNIST
Maneja la lectura de archivos binarios .gz del dataset MNIST
"""

import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def cargar_imagenes(nombre_archivo):
    """
    Carga imágenes desde archivo IDX3-UBYTE comprimido con gzip.
    
    Formato MNIST:
    - 4 bytes: Magic number (2051)
    - 4 bytes: Número de imágenes
    - 4 bytes: Número de filas (28)
    - 4 bytes: Número de columnas (28)
    - Datos: píxeles en formato big-endian
    
    Returns:
        numpy.ndarray: Array de forma (N, 28, 28) con valores 0-255
    """
    with gzip.open(nombre_archivo, 'rb') as f:
        # Leer cabecera (big-endian, 4 enteros sin signo)
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        
        if magic != 2051:
            raise ValueError(f"Archivo inválido. Magic number esperado: 2051, recibido: {magic}")
        
        # Leer píxeles y convertir a array NumPy
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        
        # Reorganizar: (num_imagenes, 28, 28)
        data = data.reshape(num, rows, cols)
        return data

def cargar_etiquetas(nombre_archivo):
    """
    Carga etiquetas desde archivo IDX1-UBYTE comprimido con gzip.
    
    Formato MNIST:
    - 4 bytes: Magic number (2049)
    - 4 bytes: Número de items
    - Datos: etiquetas (0-9)
    
    Returns:
        numpy.ndarray: Array de forma (N,) con valores 0-9
    """
    with gzip.open(nombre_archivo, 'rb') as f:
        # Leer cabecera
        magic, num = struct.unpack(">II", f.read(8))
        
        if magic != 2049:
            raise ValueError(f"Archivo inválido. Magic number esperado: 2049, recibido: {magic}")
        
        # Leer etiquetas
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        
        return data

# ============================================================================
# PROGRAMA DE PRUEBA
# ============================================================================

if __name__ == "__main__":
    print("=== Verificación del Dataset MNIST ===\n")

    try:
        # Obtener rutas a los archivos
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, '..', 'data')
        
        # Cargar datasets de entrenamiento y prueba
        print("Cargando datos...")
        x_train = cargar_imagenes(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
        y_train = cargar_etiquetas(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
        x_test = cargar_imagenes(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
        y_test = cargar_etiquetas(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

        print("\n✅ Lectura exitosa!")
        print(f"Dataset Entrenamiento: {x_train.shape} imágenes")
        print(f"Dataset Prueba:        {x_test.shape} imágenes")

        # Visualizar imagen aleatoria para verificación
        indice_aleatorio = np.random.randint(0, 60000)
        imagen = x_train[indice_aleatorio]
        etiqueta = y_train[indice_aleatorio]

        print(f"\nMostrando imagen #{indice_aleatorio} (Etiqueta: {etiqueta})")
        
        plt.imshow(imagen, cmap='gray')
        plt.title(f"Etiqueta Real: {etiqueta}")
        plt.axis('off')
        plt.show()

    except FileNotFoundError:
        print("❌ ERROR: No se encuentran los archivos .gz en la carpeta '../data/'")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    