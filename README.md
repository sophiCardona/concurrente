# Red Neuronal MLP - Clasificación MNIST

Este proyecto implementa una red neuronal perceptrón multicapa (MLP) para clasificar dígitos escritos a mano del dataset MNIST. Se desarrollaron **5 versiones diferentes** para comparar el rendimiento entre implementaciones secuenciales y paralelas en Python y C, incluyendo aceleración GPU.

## 📋 Descripción del Proyecto

La red neuronal implementada tiene la siguiente arquitectura:
- **Capa de entrada:** 784 neuronas (imágenes 28x28 píxeles)
- **Capa oculta:** 512 neuronas con activación ReLU
- **Capa de salida:** 10 neuronas con activación Softmax (dígitos 0-9)

El objetivo es entrenar la red para reconocer dígitos y comparar los tiempos de ejecución entre diferentes enfoques de implementación.

---

## Versiones Implementadas

### 1. Python Secuencial (`python_secuencial/`)

Implementación base en Python usando NumPy. Procesamiento completamente secuencial.

**Ejecutar:**
```bash
cd python_secuencial
python entrenamiento.py
```

**Archivos principales:**
- `verificar_mnist.py`: Carga los datos MNIST desde archivos `.gz`
- `preprocesamiento.py`: Normalización y codificación one-hot
- `entrenamiento.py`: Entrenamiento secuencial de la red

---

### 2. C Secuencial (`c_secuencial/`)

Implementación en C puro sin optimizaciones de paralelismo. Usa álgebra lineal manual.

**Compilar y ejecutar:**
```bash
cd c_secuencial
gcc mlp.c -o mlp.exe -O3
./mlp.exe
```

**Optimizaciones:** `-O3` para optimización del compilador

---

### 3. Python con Multiprocessing (`python_mp/`)

Versión paralela usando `multiprocessing` de Python. Divide el dataset en lotes y los procesa en múltiples cores.

**Ejecutar:**
```bash
cd python_mp
python mp_entrenamiento.py
```

**Características:**
- Divide el entrenamiento entre múltiples procesos
- Cada proceso calcula gradientes en paralelo
- Proceso principal agrega los gradientes y actualiza pesos

---

### 4. C con OpenMP (`c_openmp/`)

Implementación en C usando OpenMP para paralelización automática de bucles críticos.

**Compilar y ejecutar:**
```bash
cd c_openmp
gcc mlp.c -o mlp_omp.exe -O3 -fopenmp
./mlp_omp.exe
```

**Características:**
- Paralelización de operaciones matriciales con `#pragma omp parallel for`
- Control del número de hilos con `OMP_NUM_THREADS`

---

### 5. C con CUDA (`c_cuda/`)

Implementación GPU en C usando CUDA para acelerar operaciones matriciales en la tarjeta gráfica NVIDIA.

**⚠️ Requisitos:**
- **Tarjeta gráfica NVIDIA** con soporte CUDA
- NVIDIA CUDA Toolkit instalado
- Compilador `nvcc` disponible en PATH

**Compilar y ejecutar:**
```bash
cd c_cuda
nvcc mlp_cuda.cu -o mlp_cuda.exe -O3
./mlp_cuda.exe
```

**Características:**
- Kernels CUDA personalizados para multiplicación de matrices
- Operaciones de forward/backward propagation en GPU
- Transferencia de datos CPU ↔ GPU optimizada
- Aprovecha miles de cores CUDA en paralelo
- Máximo rendimiento con hardware especializado

---

## 📁 Estructura del Proyecto

```
proyecto/
├── data/                      # Dataset MNIST (.gz)
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
├── python_secuencial/         # Versión Python secuencial
├── python_mp/                 # Versión Python con multiprocessing
├── c_secuencial/              # Versión C secuencial
├── c_openmp/                  # Versión C con OpenMP
└── c_cuda/                    # Versión C con CUDA (GPU)
```

---

## ⚙️ Requisitos

**Python:**
- Python 3.8+
- NumPy
- Matplotlib (para visualización)

**C:**
- GCC (MinGW en Windows)
- Soporte OpenMP

**C con CUDA (opcional):**
- Tarjeta gráfica NVIDIA con soporte CUDA
- NVIDIA CUDA Toolkit
- Compilador `nvcc`

---

## 📊 Comparación de Rendimiento

Cada versión imprime el tiempo de ejecución al finalizar. Los resultados esperados (de más lento a más rápido):
1. **Python Secuencial:** Baseline (más lento)
2. **C Secuencial:** ~3-5x más rápido que Python secuencial
3. **Python MP:** Speedup según número de cores (2-4x en CPU de 4-8 cores)
4. **C OpenMP:** Máximo rendimiento en CPU (~5-10x)
5. **C CUDA:** Máximo rendimiento absoluto con GPU NVIDIA (10-100x según GPU)