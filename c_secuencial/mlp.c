/*
 * MLP Neural Network - Implementación Secuencial en C
 * Red neuronal de 3 capas (Entrada -> Oculta -> Salida) para clasificación MNIST
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// ============================================================================
// SECCIÓN 1: ESTRUCTURA DE DATOS - MATRIZ
// ============================================================================

typedef struct {
    double *data;  // Datos almacenados en formato fila-mayor (row-major)
    int rows;      // Número de filas
    int cols;      // Número de columnas
} Matrix;

// Crear matriz inicializada en ceros
Matrix* matrix_create(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    size_t size = (size_t)rows * (size_t)cols;
    m->data = (double*)calloc(size, sizeof(double));
    return m;
}

// Liberar memoria de una matriz
void matrix_free(Matrix *m) {
    if (m != NULL) {
        if (m->data != NULL) free(m->data);
        free(m);
    }
}

// Inicializar matriz con valores aleatorios uniformes en [-scale, scale]
void matrix_randomize(Matrix *m, double scale) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * scale;
    }
}

// ============================================================================
// SECCIÓN 2: OPERACIONES MATRICIALES
// ============================================================================

// Multiplicación de matrices: C = A × B
// Complejidad: O(m × n × p) donde A es m×n y B es n×p
Matrix* matrix_multiply(Matrix *A, Matrix *B) {
    if (A->cols != B->rows) {
        printf("Error: Dimensiones incompatibles %dx%d * %dx%d\n", 
               A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }
    
    Matrix *C = matrix_create(A->rows, B->cols);
    
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A->cols; k++) {
                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
            C->data[i * C->cols + j] = sum;
        }
    }
    return C;
}

// Transpuesta de matriz: T = A^T
Matrix* matrix_transpose(Matrix *A) {
    Matrix *T = matrix_create(A->cols, A->rows);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            T->data[j * T->cols + i] = A->data[i * A->cols + j];
        }
    }
    return T;
}

// Sumar vector de bias a cada fila de la matriz (broadcasting)
void matrix_add_bias(Matrix *A, Matrix *b) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            A->data[i * A->cols + j] += b->data[j];
        }
    }
}

// ============================================================================
// SECCIÓN 3: FUNCIONES DE ACTIVACIÓN
// ============================================================================

// ReLU: f(x) = max(0, x)
void apply_relu(Matrix *M) {
    for (int i = 0; i < M->rows * M->cols; i++) {
        if (M->data[i] < 0) M->data[i] = 0;
    }
}

// Derivada de ReLU: f'(x) = 1 si x > 0, 0 en caso contrario
Matrix* relu_derivative(Matrix *Z) {
    Matrix *D = matrix_create(Z->rows, Z->cols);
    for (int i = 0; i < Z->rows * Z->cols; i++) {
        D->data[i] = (Z->data[i] > 0) ? 1.0 : 0.0;
    }
    return D;
}

// Softmax: Convierte logits en probabilidades normalizadas
// f(x_i) = exp(x_i) / Σ exp(x_j)
void apply_softmax(Matrix *M) {
    for (int i = 0; i < M->rows; i++) {
        // Encontrar valor máximo para estabilidad numérica
        double max_val = -1e9;
        for (int j = 0; j < M->cols; j++) {
            double val = M->data[i * M->cols + j];
            if (val > max_val) max_val = val;
        }
        
        // Calcular exponenciales y su suma
        double sum = 0.0;
        for (int j = 0; j < M->cols; j++) {
            double val = exp(M->data[i * M->cols + j] - max_val);
            M->data[i * M->cols + j] = val;
            sum += val;
        }
        
        // Normalizar para obtener probabilidades
        for (int j = 0; j < M->cols; j++) {
            M->data[i * M->cols + j] /= sum;
        }
    }
}

// ============================================================================
// SECCIÓN 4: UTILIDADES AUXILIARES
// ============================================================================

// Resta elemento a elemento: A = A - B
void matrix_subtract(Matrix *A, Matrix *B) {
    for (int i = 0; i < A->rows * A->cols; i++) {
        A->data[i] -= B->data[i];
    }
}

// Convertir etiquetas a formato one-hot encoding
// Ejemplo: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Matrix* to_one_hot(unsigned char *labels, int rows, int num_classes) {
    Matrix *Y = matrix_create(rows, num_classes);
    for (int i = 0; i < rows; i++) {
        Y->data[i * num_classes + (int)labels[i]] = 1.0;
    }
    return Y;
}

// Obtener índice del valor máximo en una fila (predicción de la red)
int get_argmax(Matrix *M, int row) {
    double max_val = -1e9;
    int max_idx = 0;
    for (int j = 0; j < M->cols; j++) {
        if (M->data[row * M->cols + j] > max_val) {
            max_val = M->data[row * M->cols + j];
            max_idx = j;
        }
    }
    return max_idx;
}

// Leer entero de 4 bytes en formato big-endian (formato MNIST)
int read_int(FILE *fp) {
    unsigned char buf[4];
    if (fread(buf, 1, 4, fp) != 4) return 0;
    return (int)buf[0] << 24 | (int)buf[1] << 16 | (int)buf[2] << 8 | (int)buf[3];
}

// ============================================================================
// SECCIÓN 5: RED NEURONAL MLP
// ============================================================================

typedef struct {
    // Pesos y biases
    Matrix *W1, *b1;  // Capa oculta
    Matrix *W2, *b2;  // Capa de salida
    
    // Cache de activaciones (se reutilizan durante entrenamiento)
    Matrix *A1, *Z1;  // Capa oculta
    Matrix *A2, *Z2;  // Capa de salida
} MLP;

// Crear red neuronal con inicialización aleatoria de pesos
MLP* mlp_create(int input, int hidden, int output) {
    MLP *net = (MLP*)malloc(sizeof(MLP));
    
    // Inicializar pesos y biases
    net->W1 = matrix_create(input, hidden);
    net->b1 = matrix_create(1, hidden);
    net->W2 = matrix_create(hidden, output);
    net->b2 = matrix_create(1, output);
    
    matrix_randomize(net->W1, 0.1);
    matrix_randomize(net->W2, 0.1);
    
    // Cache inicia vacío
    net->A1 = NULL;
    net->Z1 = NULL;
    net->A2 = NULL;
    net->Z2 = NULL;
    
    return net;
}

// Forward Propagation: Calcular predicciones de la red
// Z1 = X·W1 + b1  →  A1 = ReLU(Z1)
// Z2 = A1·W2 + b2 →  A2 = Softmax(Z2)
void mlp_forward(MLP *net, Matrix *X) {
    // Limpiar cache anterior
    if (net->Z1) matrix_free(net->Z1);
    if (net->A1) matrix_free(net->A1);
    if (net->Z2) matrix_free(net->Z2);
    if (net->A2) matrix_free(net->A2);

    // Capa oculta: Z1 = X·W1 + b1, A1 = ReLU(Z1)
    net->Z1 = matrix_multiply(X, net->W1);
    matrix_add_bias(net->Z1, net->b1);
    
    net->A1 = matrix_create(net->Z1->rows, net->Z1->cols);
    memcpy(net->A1->data, net->Z1->data, 
           net->Z1->rows * net->Z1->cols * sizeof(double));
    apply_relu(net->A1);

    // Capa de salida: Z2 = A1·W2 + b2, A2 = Softmax(Z2)
    net->Z2 = matrix_multiply(net->A1, net->W2);
    matrix_add_bias(net->Z2, net->b2);
    
    net->A2 = matrix_create(net->Z2->rows, net->Z2->cols);
    memcpy(net->A2->data, net->Z2->data, 
           net->Z2->rows * net->Z2->cols * sizeof(double));
    apply_softmax(net->A2);
}

// Backward Propagation: Calcular gradientes y actualizar pesos
// Implementa el algoritmo de retropropagación estándar
void mlp_backward(MLP *net, Matrix *X, Matrix *Y, double lr) {
    int m = X->rows;  // Tamaño del batch

    // ---- Gradientes de la capa de salida ----
    
    // Error de salida: dZ2 = A2 - Y (derivada de cross-entropy + softmax)
    Matrix *dZ2 = matrix_create(net->A2->rows, net->A2->cols);
    memcpy(dZ2->data, net->A2->data, 
           net->A2->rows * net->A2->cols * sizeof(double));
    for(int i = 0; i < dZ2->rows * dZ2->cols; i++) {
        dZ2->data[i] -= Y->data[i];
    }

    // Gradiente de W2: dW2 = A1^T · dZ2
    Matrix *A1_T = matrix_transpose(net->A1);
    Matrix *dW2 = matrix_multiply(A1_T, dZ2);

    // Gradiente de b2: db2 = suma por columnas de dZ2
    Matrix *db2 = matrix_create(1, net->b2->cols);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < dZ2->cols; j++) {
            db2->data[j] += dZ2->data[i * dZ2->cols + j];
        }
    }

    // ---- Gradientes de la capa oculta ----
    
    // Propagación del error: dA1 = dZ2 · W2^T
    Matrix *W2_T = matrix_transpose(net->W2);
    Matrix *dA1 = matrix_multiply(dZ2, W2_T);
    
    // Aplicar derivada de ReLU: dZ1 = dA1 ⊙ ReLU'(Z1)
    Matrix *dRe = relu_derivative(net->Z1);
    Matrix *dZ1 = matrix_create(dA1->rows, dA1->cols);
    for(int i = 0; i < dA1->rows * dA1->cols; i++) {
        dZ1->data[i] = dA1->data[i] * dRe->data[i];
    }

    // Gradiente de W1: dW1 = X^T · dZ1
    Matrix *X_T = matrix_transpose(X);
    Matrix *dW1 = matrix_multiply(X_T, dZ1);

    // Gradiente de b1: db1 = suma por columnas de dZ1
    Matrix *db1 = matrix_create(1, net->b1->cols);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < dZ1->cols; j++) {
            db1->data[j] += dZ1->data[i * dZ1->cols + j];
        }
    }

    // ---- Actualización de parámetros (Gradient Descent) ----
    
    double scalar = lr / m;  // Learning rate normalizado por tamaño de batch
    
    // Actualizar W1 y b1
    for(int i = 0; i < net->W1->rows * net->W1->cols; i++) {
        net->W1->data[i] -= scalar * dW1->data[i];
    }
    for(int i = 0; i < net->b1->cols; i++) {
        net->b1->data[i] -= scalar * db1->data[i];
    }
    
    // Actualizar W2 y b2
    for(int i = 0; i < net->W2->rows * net->W2->cols; i++) {
        net->W2->data[i] -= scalar * dW2->data[i];
    }
    for(int i = 0; i < net->b2->cols; i++) {
        net->b2->data[i] -= scalar * db2->data[i];
    }

    // Liberar memoria temporal
    matrix_free(dZ2);
    matrix_free(A1_T);
    matrix_free(dW2);
    matrix_free(db2);
    matrix_free(W2_T);
    matrix_free(dA1);
    matrix_free(dRe);
    matrix_free(dZ1);
    matrix_free(X_T);
    matrix_free(dW1);
    matrix_free(db1);
}

// ============================================================================
// SECCIÓN 6: PROGRAMA PRINCIPAL
// ============================================================================

int main() {
    srand(42);  // Semilla fija para reproducibilidad

    // Rutas de datos MNIST (archivos descomprimidos)
    const char *img_path = "../data/train-images.idx3-ubyte";
    const char *lbl_path = "../data/train-labels.idx1-ubyte";

    printf("=== MLP Secuencial en C ===\n\n");

    // ---- Cargar dataset MNIST ----
    
    FILE *f_img = fopen(img_path, "rb");
    FILE *f_lbl = fopen(lbl_path, "rb");
    
    if (!f_img || !f_lbl) {
        printf("Error: No se encuentran los archivos en ../data/\n");
        printf("Asegúrate de que NO tengan extensión .gz (descomprímelos)\n");
        return 1;
    }

    // Leer headers del formato IDX
    read_int(f_img);  // Magic number
    int num_imgs = read_int(f_img);
    int rows = read_int(f_img);
    int cols = read_int(f_img);
    read_int(f_lbl);  // Magic number
    read_int(f_lbl);  // Número de items

    int LIMIT = 60000;  // Usar todo el dataset de entrenamiento
    printf("Cargando %d imágenes de %dx%d...\n", LIMIT, rows, cols);

    // Reservar memoria para datos
    Matrix *X_train = matrix_create(LIMIT, rows * cols);
    unsigned char *y_temp = (unsigned char*)malloc(LIMIT);

    // Leer y normalizar píxeles [0-255] -> [0.0-1.0]
    for (int i = 0; i < LIMIT; i++) {
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel;
            fread(&pixel, 1, 1, f_img);
            X_train->data[i * X_train->cols + j] = (double)pixel / 255.0;
        }
    }
    
    // Leer etiquetas
    fread(y_temp, 1, LIMIT, f_lbl);
    
    // Convertir a one-hot encoding
    Matrix *Y_train = to_one_hot(y_temp, LIMIT, 10);
    
    fclose(f_img);
    fclose(f_lbl);
    printf("Datos cargados exitosamente.\n\n");

    // ---- Configurar red neuronal ----
    
    int INPUT_SIZE = 784;    // 28x28 píxeles
    int HIDDEN_SIZE = 256;   // Neuronas en capa oculta
    int OUTPUT_SIZE = 10;    // 10 clases (dígitos 0-9)
    
    MLP *mlp = mlp_create(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Hiperparámetros
    int EPOCHS = 10;
    int BATCH_SIZE = 64;
    double LR = 0.1;

    printf("Arquitectura: %d -> %d -> %d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Hiperparámetros: LR=%.2f, Batch=%d, Epochs=%d\n\n", LR, BATCH_SIZE, EPOCHS);
    printf("Iniciando entrenamiento...\n");
    
    clock_t start = clock();

    // ---- Bucle de entrenamiento ----
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        int correct = 0;
        
        // Iterar sobre mini-batches
        for (int i = 0; i < LIMIT; i += BATCH_SIZE) {
            int current_batch = (LIMIT - i < BATCH_SIZE) ? (LIMIT - i) : BATCH_SIZE;
            
            // Crear mini-batch temporal
            Matrix *X_batch = matrix_create(current_batch, INPUT_SIZE);
            Matrix *Y_batch = matrix_create(current_batch, OUTPUT_SIZE);
            
            // Copiar datos del batch
            for(int r = 0; r < current_batch; r++) {
                memcpy(&X_batch->data[r * INPUT_SIZE], 
                       &X_train->data[(i + r) * INPUT_SIZE], 
                       INPUT_SIZE * sizeof(double));
                memcpy(&Y_batch->data[r * OUTPUT_SIZE],  
                       &Y_train->data[(i + r) * OUTPUT_SIZE],  
                       OUTPUT_SIZE * sizeof(double));
            }

            // Forward + Backward propagation
            mlp_forward(mlp, X_batch);
            mlp_backward(mlp, X_batch, Y_batch, LR);

            // Calcular accuracy del batch
            for(int r = 0; r < current_batch; r++) {
                if (get_argmax(mlp->A2, r) == (int)y_temp[i + r]) {
                    correct++;
                }
            }

            // Liberar memoria del batch
            matrix_free(X_batch);
            matrix_free(Y_batch);
        }
        
        // Reportar progreso por época
        double accuracy = (double)correct / LIMIT * 100.0;
        printf("Época %2d/%d | Accuracy: %.2f%%\n", epoch + 1, EPOCHS, accuracy);
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\n=== Entrenamiento Completado ===\n");
    printf("Tiempo total: %.2f segundos\n", time_taken);

    // ---- Limpieza de memoria ----
    
    free(y_temp);
    matrix_free(X_train);
    matrix_free(Y_train);
    
    // Liberar red neuronal (falta implementar mlp_free completo)
    matrix_free(mlp->W1);
    matrix_free(mlp->b1);
    matrix_free(mlp->W2);
    matrix_free(mlp->b2);
    if (mlp->Z1) matrix_free(mlp->Z1);
    if (mlp->A1) matrix_free(mlp->A1);
    if (mlp->Z2) matrix_free(mlp->Z2);
    if (mlp->A2) matrix_free(mlp->A2);
    free(mlp);
    
    return 0;
}