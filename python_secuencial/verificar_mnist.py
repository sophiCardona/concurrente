import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import os

def cargar_imagenes(nombre_archivo):
    with gzip.open(nombre_archivo, 'rb') as f:
        # Leer el encabezado: magic number, num_images, rows, cols
        # >IIII significa: big-endian, 4 enteros sin signo
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))        

        if magic != 2051:  
            raise ValueError(f"El archivo no es un archivo de imágenes MNIST válido.Magic number incorrecto: {magic}.")      
        # Leer los datos de las imágenes
        buffer = f.read()

        # Convertir a array de numpy (enteros de 0 a 255)
        data = np.frombuffer(buffer, dtype=np.uint8)
        
        # Reorganizar: (Num Imagenes, 28, 28)
        data = data.reshape(num, rows, cols)
        return data

def cargar_etiquetas(nombre_archivo):
    with gzip.open(nombre_archivo, 'rb') as f:
        # Leer cabecera: magic number, num_items
        magic, num = struct.unpack(">II", f.read(8))
        
        # Verificar que es el archivo de etiquetas (magic number 2049)
        if magic != 2049:
            raise ValueError(f"Magic number incorrecto: {magic}. ¿Es el archivo de etiquetas?")
            
        buffer = f.read()
       # Convertir a array de numpy (enteros de 0 a 255)
        data = np.frombuffer(buffer, dtype=np.uint8)
        
        return data

# --- EJECUCIÓN ---
if __name__ == "__main__":
    print("Cargando datos a mano...")

    try:
        # Obtener la ruta al directorio data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, '..', 'data')
        
        # 1. Cargar Entrenamiento
        x_train = cargar_imagenes(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
        y_train = cargar_etiquetas(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
        
        # 2. Cargar Prueba
        x_test = cargar_imagenes(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
        y_test = cargar_etiquetas(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

        print("\n¡Lectura Exitosa!")
        print(f"Dataset Entrenamiento: {x_train.shape} imágenes") # Debe decir (60000, 28, 28)
        print(f"Dataset Prueba:        {x_test.shape} imágenes")  # Debe decir (10000, 28, 28)

        # 3. Visualizar una imagen aleatoria para confirmar
        indice_aleatorio = np.random.randint(0, 60000)
        imagen = x_train[indice_aleatorio]
        etiqueta = y_train[indice_aleatorio]

        print(f"\nMostrando imagen del índice {indice_aleatorio}. Debería ser un número: {etiqueta}")
        
        plt.imshow(imagen, cmap='gray')
        plt.title(f"Etiqueta Real: {etiqueta}")
        plt.axis('off') # Quitar ejes para que se vea limpio
        plt.show()

    except FileNotFoundError:
        print("❌ ERROR: No encuentro los archivos .gz. Asegúrate de que estén en la misma carpeta que este script.")
    except Exception as e:
        print(f"❌ ERROR al leer: {e}")
    