import numpy as np
from verificar_mnist import cargar_imagenes, cargar_etiquetas

def one_hot_encode(etiquetas, num_clases=10):
    n = etiquetas.shape[0]
    one_hot = np.zeros((n, num_clases))
    one_hot[np.arange(n), etiquetas] = 1
    return one_hot

def preprocesar_datos(x, y):
    num_imagenes = x.shape[0]
    x_plano = x.reshape(num_imagenes, 784)
    x_norm = x_plano / 255.0
    y_encoded = one_hot_encode(y)
    return x_norm, y_encoded

# --- FUNCIÓN MAESTRA ---
def obtener_datos_listos():
    """
    Esta función la llamarás desde tu archivo de entrenamiento.
    Se encarga de todo el trabajo sucio y te devuelve los datos limpios.
    """
    print("--- Iniciando carga de datos ---")
    x_train = cargar_imagenes('train-images-idx3-ubyte.gz')
    y_train = cargar_etiquetas('train-labels-idx1-ubyte.gz')
    x_test = cargar_imagenes('t10k-images-idx3-ubyte.gz')
    y_test = cargar_etiquetas('t10k-labels-idx1-ubyte.gz')
    
    print("--- Preprocesando datos ---")
    x_train_final, y_train_final = preprocesar_datos(x_train, y_train)
    x_test_final, y_test_final = preprocesar_datos(x_test, y_test)
    
    return x_train_final, y_train_final, x_test_final, y_test_final

# --- EJECUCIÓN DE PRUEBA ---
if __name__ == "__main__":
    # Ahora el main solo prueba la función maestra
    x_tr, y_tr, x_te, y_te = obtener_datos_listos()

    print(f"Forma final entrada:   {x_tr.shape}")
    print(f"Forma etiqueta one-hot:  {y_tr[0]}")
    print("¡Todo listo para el entrenamiento!")