#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>

// --- 1. ESTRUCTURA DE MATRIZ Y GESTIÓN DE MEMORIA ---

typedef struct {
    double *data;
    int rows;
    int cols;
} Matrix;

// Crear una matriz vacía (llena de ceros)
// Versión corregida de matrix_create
Matrix* matrix_create(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    
    // CAMBIO AQUÍ: Convertimos a size_t antes de multiplicar
    size_t size = (size_t)rows * (size_t)cols;
    
    m->data = (double*)calloc(size, sizeof(double)); 
    return m;
}

// Liberar memoria (IMPORTANTE EN C)
void matrix_free(Matrix *m) {
    if (m != NULL) {
        if (m->data != NULL) free(m->data);
        free(m);
    }
}

// Inicializar con valores aleatorios (He/Xavier simplificado)
void matrix_randomize(Matrix *m, double scale) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * scale;
    }
}

// Acceso rápido: M[row][col]
// Se usa macro o funcion inline para limpieza, aquí manual: m->data[r * cols + c]

// --- 2. OPERACIONES MATEMÁTICAS MANUALES ---

// Multiplicación C = A * B
Matrix* matrix_multiply(Matrix *A, Matrix *B) {
    if (A->cols != B->rows) {
        printf("Error dimensión: %dx%d * %dx%d\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }
    Matrix *C = matrix_create(A->rows, B->cols);
    
    #pragma omp parallel for
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A->cols; k++) {
                // A[i][k] * B[k][j]
                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
            C->data[i * C->cols + j] = sum;
        }
    }
    return C;
}

// Transpuesta
Matrix* matrix_transpose(Matrix *A) {
    Matrix *T = matrix_create(A->cols, A->rows);
    
    #pragma omp parallel for
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            T->data[j * T->cols + i] = A->data[i * A->cols + j];
        }
    }
    return T;
}

// Sumar Bias (Broadcasting)
void matrix_add_bias(Matrix *A, Matrix *b) {
    #pragma omp parallel for
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            A->data[i * A->cols + j] += b->data[j];
        }
    }
}

// ReLU: max(0, x)
void apply_relu(Matrix *M) {
    #pragma omp parallel for
    for (int i = 0; i < M->rows * M->cols; i++) {
        if (M->data[i] < 0) M->data[i] = 0;
    }
}

// Derivada ReLU
Matrix* relu_derivative(Matrix *Z) {
    Matrix *D = matrix_create(Z->rows, Z->cols);
    #pragma omp parallel for
    for (int i = 0; i < Z->rows * Z->cols; i++) {
        D->data[i] = (Z->data[i] > 0) ? 1.0 : 0.0;
    }
    return D;
}

// Softmax
void apply_softmax(Matrix *M) {
    #pragma omp parallel for
    for (int i = 0; i < M->rows; i++) {
        // 1. Encontrar maximo (estabilidad)
        double max_val = -1e9;
        for (int j = 0; j < M->cols; j++) {
            double val = M->data[i * M->cols + j];
            if (val > max_val) max_val = val;
        }
        
        // 2. Exponenciales
        double sum = 0.0;
        for (int j = 0; j < M->cols; j++) {
            double val = exp(M->data[i * M->cols + j] - max_val);
            M->data[i * M->cols + j] = val;
            sum += val;
        }
        
        // 3. Normalizar
        for (int j = 0; j < M->cols; j++) {
            M->data[i * M->cols + j] /= sum;
        }
    }
}

// Resta: A = A - B (usado para gradiente)
void matrix_subtract(Matrix *A, Matrix *B) {
    for (int i = 0; i < A->rows * A->cols; i++) {
        A->data[i] -= B->data[i];
    }
}

// One Hot Helper
Matrix* to_one_hot(unsigned char *labels, int rows, int num_classes) {
    Matrix *Y = matrix_create(rows, num_classes);
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        Y->data[i * num_classes + (int)labels[i]] = 1.0;
    }
    return Y;
}

// Argmax
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

// --- 3. LECTURA DE ARCHIVOS MNIST (Big Endian) ---

int read_int(FILE *fp) {
    unsigned char buf[4];
    if (fread(buf, 1, 4, fp) != 4) return 0;
    return (int)buf[0] << 24 | (int)buf[1] << 16 | (int)buf[2] << 8 | (int)buf[3];
}

// --- 4. ESTRUCTURA MLP Y ENTRENAMIENTO ---

typedef struct {
    Matrix *W1, *b1, *W2, *b2;
    // Cache de activaciones (se sobrescriben en cada forward)
    Matrix *A1, *Z1, *A2, *Z2; 
} MLP;

MLP* mlp_create(int input, int hidden, int output) {
    MLP *net = (MLP*)malloc(sizeof(MLP));
    net->W1 = matrix_create(input, hidden);
    net->b1 = matrix_create(1, hidden);
    net->W2 = matrix_create(hidden, output);
    net->b2 = matrix_create(1, output);
    
    matrix_randomize(net->W1, 0.1);
    matrix_randomize(net->W2, 0.1);
    
    // Inicializar punteros de cache en NULL
    net->A1 = NULL; net->Z1 = NULL; net->A2 = NULL; net->Z2 = NULL;
    return net;
}

void mlp_forward(MLP *net, Matrix *X) {
    // Limpiar memoria anterior si existe
    if (net->Z1) matrix_free(net->Z1);
    if (net->A1) matrix_free(net->A1);
    if (net->Z2) matrix_free(net->Z2);
    if (net->A2) matrix_free(net->A2);

    // Capa 1
    net->Z1 = matrix_multiply(X, net->W1);
    matrix_add_bias(net->Z1, net->b1);
    
    net->A1 = matrix_create(net->Z1->rows, net->Z1->cols);
    #pragma omp parallel for
    for(int i=0; i<net->Z1->rows * net->Z1->cols; i++) net->A1->data[i] = net->Z1->data[i];
    apply_relu(net->A1);

    // Capa 2
    net->Z2 = matrix_multiply(net->A1, net->W2);
    matrix_add_bias(net->Z2, net->b2);
    
    net->A2 = matrix_create(net->Z2->rows, net->Z2->cols);
    #pragma omp parallel for
    for(int i=0; i<net->Z2->rows * net->Z2->cols; i++) net->A2->data[i] = net->Z2->data[i];
    apply_softmax(net->A2);
}

void mlp_backward(MLP *net, Matrix *X, Matrix *Y, double lr) {
    int m = X->rows;

    // 1. Error Salida: dZ2 = A2 - Y
    Matrix *dZ2 = matrix_create(net->A2->rows, net->A2->cols);
    #pragma omp parallel for
    for(int i=0; i<dZ2->rows * dZ2->cols; i++) dZ2->data[i] = net->A2->data[i] - Y->data[i];

    // 2. Gradiente W2: A1.T * dZ2
    Matrix *A1_T = matrix_transpose(net->A1);
    Matrix *dW2 = matrix_multiply(A1_T, dZ2);

    // 3. Gradiente b2: Suma de columnas de dZ2
    Matrix *db2 = matrix_create(1, net->b2->cols);
    for(int i=0; i<m; i++) {
        for(int j=0; j<dZ2->cols; j++) {
            db2->data[j] += dZ2->data[i * dZ2->cols + j];
        }
    }

    // 4. Error Oculta: dA1 = dZ2 * W2.T
    Matrix *W2_T = matrix_transpose(net->W2);
    Matrix *dA1 = matrix_multiply(dZ2, W2_T);
    
    // dZ1 = dA1 * relu_deriv(Z1)
    Matrix *dRe = relu_derivative(net->Z1);
    Matrix *dZ1 = matrix_create(dA1->rows, dA1->cols);

    #pragma omp parallel for
    for(int i=0; i<dA1->rows * dA1->cols; i++) {
        dZ1->data[i] = dA1->data[i] * dRe->data[i];
    }

    // 5. Gradiente W1: X.T * dZ1
    Matrix *X_T = matrix_transpose(X);
    Matrix *dW1 = matrix_multiply(X_T, dZ1);

    Matrix *db1 = matrix_create(1, net->b1->cols);
    for(int i=0; i<m; i++) {
        for(int j=0; j<dZ1->cols; j++) {
            db1->data[j] += dZ1->data[i * dZ1->cols + j];
        }
    }

    // 6. Actualización de Pesos
    double scalar = lr / m;
    
    // W1 = W1 - scalar * dW1
    #pragma omp parallel for
    for(int i=0; i<net->W1->rows * net->W1->cols; i++) net->W1->data[i] -= scalar * dW1->data[i];
    #pragma omp parallel for
    for(int i=0; i<net->b1->cols; i++) net->b1->data[i] -= scalar * db1->data[i];
    #pragma omp parallel for
    for(int i=0; i<net->W2->rows * net->W2->cols; i++) net->W2->data[i] -= scalar * dW2->data[i];
    #pragma omp parallel for
    for(int i=0; i<net->b2->cols; i++) net->b2->data[i] -= scalar * db2->data[i];

    // Limpieza de memoria temporal (¡Crucial en C!)
    matrix_free(dZ2); matrix_free(A1_T); matrix_free(dW2); matrix_free(db2);
    matrix_free(W2_T); matrix_free(dA1); matrix_free(dRe); matrix_free(dZ1);
    matrix_free(X_T); matrix_free(dW1); matrix_free(db1);
}

// --- 5. MAIN ---

int main() {
    srand(42); 

    const char *img_path = "../data/train-images.idx3-ubyte";
    const char *lbl_path = "../data/train-labels.idx1-ubyte";

    // <--- 2. DETECTAR NÚCLEOS (Opcional pero recomendado para verificar)
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads); // Forzar uso de todos
    printf("--- INICIANDO OPENMP CON %d HILOS ---\n", max_threads);

    // 1. Cargar Datos (Esto se queda igual, es lectura de disco secuencial)
    FILE *f_img = fopen(img_path, "rb");
    FILE *f_lbl = fopen(lbl_path, "rb");
    
    if (!f_img || !f_lbl) {
        printf("Error: No encuentro los archivos en ../data/\n");
        return 1;
    }

    read_int(f_img); 
    int num_imgs = read_int(f_img);
    int rows = read_int(f_img);
    int cols = read_int(f_img);
    read_int(f_lbl); 
    read_int(f_lbl);

    int LIMIT = 60000; 
    printf("Cargando %d imagenes de %dx%d...\n", LIMIT, rows, cols);

    Matrix *X_train = matrix_create(LIMIT, rows * cols);
    unsigned char *y_temp = (unsigned char*)malloc(LIMIT);

    for (int i = 0; i < LIMIT; i++) {
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel;
            fread(&pixel, 1, 1, f_img);
            X_train->data[i * X_train->cols + j] = (double)pixel / 255.0;
        }
    }
    fread(y_temp, 1, LIMIT, f_lbl);
    
    // <--- 3. ONE HOT se puede paralelizar (agrega #pragma omp parallel for dentro de la función to_one_hot)
    Matrix *Y_train = to_one_hot(y_temp, LIMIT, 10);
    
    fclose(f_img);
    fclose(f_lbl);

    MLP *mlp = mlp_create(784, 256, 10);
    int EPOCHS = 10;
    int BATCH_SIZE = 64;
    double LR = 0.1;

    printf("Entrenando %d epocas...\n", EPOCHS);
    
    // <--- 4. CAMBIO DE CRONÓMETRO (CRUCIAL)
    // clock() mide tiempo de CPU (suma todos los hilos).
    // omp_get_wtime() mide tiempo real (reloj de pared).
    double start = omp_get_wtime(); 

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        int correct = 0;
        
        for (int i = 0; i < LIMIT; i += BATCH_SIZE) {
            int current_batch = (LIMIT - i < BATCH_SIZE) ? (LIMIT - i) : BATCH_SIZE;
            
            Matrix *X_batch = matrix_create(current_batch, 784);
            Matrix *Y_batch = matrix_create(current_batch, 10);
            
            // <--- 5. PARALELIZAR COPIA DE MEMORIA (Pequeña optimización)
            #pragma omp parallel for
            for(int r=0; r<current_batch; r++) {
                memcpy(&X_batch->data[r*784], &X_train->data[(i+r)*784], 784*sizeof(double));
                memcpy(&Y_batch->data[r*10],  &Y_train->data[(i+r)*10],  10*sizeof(double));
            }

            // Entrenar (Se asume que mlp_forward y mlp_backward YA TIENEN los #pragma omp dentro)
            mlp_forward(mlp, X_batch);
            mlp_backward(mlp, X_batch, Y_batch, LR);

            // Calcular Accuracy (Esto es tan rápido que paralelizarlo a veces no vale la pena, pero se puede)
            for(int r=0; r<current_batch; r++) {
                if (get_argmax(mlp->A2, r) == (int)y_temp[i+r]) correct++;
            }

            matrix_free(X_batch);
            matrix_free(Y_batch);
        }
        printf("Epoch %d | Accuracy: %.2f%%\n", epoch+1, (double)correct/LIMIT * 100.0);
    }

    // <--- 6. FIN CRONÓMETRO
    double end = omp_get_wtime();
    double time_taken = end - start;

    printf("\n--- RESULTADO OPENMP ---\n");
    printf("Tiempo: %.4f segundos\n", time_taken);

    free(y_temp);
    matrix_free(X_train);
    matrix_free(Y_train);
    
    return 0;
}