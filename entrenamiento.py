# Archivo: entrenamiento.py
import numpy as np

# Intentamos importar la función maestra
try:
    from reprocesamiento import obtener_datos_listos

except ImportError:
    print("❌ Error: No encuentro el archivo 'reprocesamiento.py' o la función.")
    print("Asegúrate de que ambos archivos estén en la misma carpeta.")
    exit()

def probar_carga():
    print("🚀 Iniciando sistema...")
    
    # 1. Traer los datos
    x_train, y_train, x_test, y_test = obtener_datos_listos()
    
    print("\n✅ Datos importados correctamente al script principal.")
    
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

if __name__ == "__main__":
    probar_carga()