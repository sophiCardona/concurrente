#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>        // <-- para clock_t, clock, CLOCKS_PER_SEC
#include <cuda_runtime.h> // Librería de CUDA

// --- MACRO PARA CHEQUEAR ERRORES CUDA ---
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        const char* msg = cudaGetErrorString(code);
        fprintf(stderr,
                "GPUassert: code=%d, msg=%s, file=%s, line=%d\n",
                (int)code,
                msg ? msg : "NULL",
                file,
                line);
        if (abort) exit(code);
    }
}

// --- ESTRUCTURAS (Usando FLOAT para velocidad en GPU) ---
typedef struct {
    float *data;
    int rows;
    int cols;
} Matrix;

Matrix* matrix_create(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (float*)calloc(rows * cols, sizeof(float));
    return m;
}

void matrix_free(Matrix *m) {
    if (m != NULL) {
        if (m->data != NULL) free(m->data);
        free(m);
    }
}

void matrix_randomize(Matrix *m, float scale) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

// --- KERNEL CUDA (Lo que corre dentro de la GPU) ---
// Cada hilo calcula UN solo elemento de la matriz de salida C
__global__ void matmul_kernel(float *A, float *B, float *C,
                              int A_rows, int A_cols, int B_cols) {
    // Calcular fila y columna basado en el ID del hilo y bloque
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        // Producto punto
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// --- FUNCIÓN ANFITRIONA (Gestiona la GPU) ---
Matrix* matrix_multiply_gpu(Matrix *A, Matrix *B) {
    if (A->cols != B->rows) {
        printf("Error dim GPU: %dx%d * %dx%d\n",
               A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    Matrix *C = matrix_create(A->rows, B->cols);

    // 1. Punteros para memoria en GPU (Device)
    float *d_A, *d_B, *d_C;
    size_t size_A = (size_t)A->rows * A->cols * sizeof(float);
    size_t size_B = (size_t)B->rows * B->cols * sizeof(float);
    size_t size_C = (size_t)C->rows * C->cols * sizeof(float);

    // 2. Reservar memoria en GPU
    cudaCheckError(cudaMalloc((void**)&d_A, size_A));
    cudaCheckError(cudaMalloc((void**)&d_B, size_B));
    cudaCheckError(cudaMalloc((void**)&d_C, size_C));

    // 3. Copiar datos: CPU (Host) -> GPU (Device)
    cudaCheckError(cudaMemcpy(d_A, A->data, size_A, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, B->data, size_B, cudaMemcpyHostToDevice));

    // 4. Configurar la cuadrícula (Grid) y Bloques
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((C->cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (C->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 5. LANZAR KERNEL
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C,
                                                  A->rows, A->cols, B->cols);
    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize()); // Esperar a que termine

    // 6. Copiar resultados: GPU (Device) -> CPU (Host)
    cudaCheckError(cudaMemcpy(C->data, d_C, size_C, cudaMemcpyDeviceToHost));

    // 7. Liberar memoria GPU
    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));

    return C;
}

// --- OPERACIONES DE APOYO (CPU) ---
Matrix* matrix_transpose(Matrix *A) {
    Matrix *T = matrix_create(A->cols, A->rows);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            T->data[j * T->cols + i] = A->data[i * A->cols + j];
        }
    }
    return T;
}

void matrix_add_bias(Matrix *A, Matrix *b) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            A->data[i * A->cols + j] += b->data[j];
        }
    }
}

void apply_relu(Matrix *M) {
    for (int i = 0; i < M->rows * M->cols; i++) {
        if (M->data[i] < 0.0f) M->data[i] = 0.0f;
    }
}

Matrix* relu_derivative(Matrix *Z) {
    Matrix *D = matrix_create(Z->rows, Z->cols);
    for (int i = 0; i < Z->rows * Z->cols; i++) {
        D->data[i] = (Z->data[i] > 0.0f) ? 1.0f : 0.0f;
    }
    return D;
}

void apply_softmax(Matrix *M) {
    for (int i = 0; i < M->rows; i++) {
        float max_val = -1e9f;
        for (int j = 0; j < M->cols; j++) {
            float v = M->data[i * M->cols + j];
            if (v > max_val) max_val = v;
        }

        float sum = 0.0f;
        for (int j = 0; j < M->cols; j++) {
            float v = M->data[i * M->cols + j];
            v = expf(v - max_val);
            M->data[i * M->cols + j] = v;
            sum += v;
        }

        for (int j = 0; j < M->cols; j++) {
            M->data[i * M->cols + j] /= sum;
        }
    }
}

Matrix* to_one_hot(unsigned char *labels, int rows, int num_classes) {
    Matrix *Y = matrix_create(rows, num_classes);
    for (int i = 0; i < rows; i++) {
        Y->data[i * num_classes + (int)labels[i]] = 1.0f;
    }
    return Y;
}

int get_argmax(Matrix *M, int row) {
    float max_val = -1e9f;
    int max_idx = 0;
    for (int j = 0; j < M->cols; j++) {
        float v = M->data[row * M->cols + j];
        if (v > max_val) {
            max_val = v;
            max_idx = j;
        }
    }
    return max_idx;
}

int read_int(FILE *fp) {
    unsigned char buf[4];
    if (fread(buf, 1, 4, fp) != 4) return 0;
    return (int)buf[0] << 24 |
           (int)buf[1] << 16 |
           (int)buf[2] << 8  |
           (int)buf[3];
}

// --- MLP ---
typedef struct {
    Matrix *W1, *b1, *W2, *b2;
    Matrix *A1, *Z1, *A2, *Z2;
} MLP;

MLP* mlp_create(int input, int hidden, int output) {
    MLP *net = (MLP*)malloc(sizeof(MLP));
    net->W1 = matrix_create(input, hidden);
    net->b1 = matrix_create(1, hidden);
    net->W2 = matrix_create(hidden, output);
    net->b2 = matrix_create(1, output);
    matrix_randomize(net->W1, 0.1f);
    matrix_randomize(net->W2, 0.1f);
    net->A1 = NULL;
    net->Z1 = NULL;
    net->A2 = NULL;
    net->Z2 = NULL;
    return net;
}

void mlp_forward(MLP *net, Matrix *X) {
    if (net->Z1) matrix_free(net->Z1);
    if (net->A1) matrix_free(net->A1);
    if (net->Z2) matrix_free(net->Z2);
    if (net->A2) matrix_free(net->A2);

    // USAMOS LA GPU AQUÍ
    net->Z1 = matrix_multiply_gpu(X, net->W1);
    matrix_add_bias(net->Z1, net->b1);

    net->A1 = matrix_create(net->Z1->rows, net->Z1->cols);
    for (int i = 0; i < net->Z1->rows * net->Z1->cols; i++) {
        net->A1->data[i] = net->Z1->data[i];
    }
    apply_relu(net->A1);

    // USAMOS LA GPU AQUÍ TAMBIÉN
    net->Z2 = matrix_multiply_gpu(net->A1, net->W2);
    matrix_add_bias(net->Z2, net->b2);

    net->A2 = matrix_create(net->Z2->rows, net->Z2->cols);
    for (int i = 0; i < net->Z2->rows * net->Z2->cols; i++) {
        net->A2->data[i] = net->Z2->data[i];
    }
    apply_softmax(net->A2);
}

void mlp_backward(MLP *net, Matrix *X, Matrix *Y, float lr) {
    int m = X->rows;

    Matrix *dZ2 = matrix_create(net->A2->rows, net->A2->cols);
    for (int i = 0; i < dZ2->rows * dZ2->cols; i++) {
        dZ2->data[i] = net->A2->data[i] - Y->data[i];
    }

    Matrix *A1_T = matrix_transpose(net->A1);
    Matrix *dW2 = matrix_multiply_gpu(A1_T, dZ2);

    Matrix *db2 = matrix_create(1, net->b2->cols);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < dZ2->cols; j++) {
            db2->data[j] += dZ2->data[i * dZ2->cols + j];
        }
    }

    Matrix *W2_T = matrix_transpose(net->W2);
    Matrix *dA1 = matrix_multiply_gpu(dZ2, W2_T);

    Matrix *dRe = relu_derivative(net->Z1);
    Matrix *dZ1 = matrix_create(dA1->rows, dA1->cols);
    for (int i = 0; i < dA1->rows * dA1->cols; i++) {
        dZ1->data[i] = dA1->data[i] * dRe->data[i];
    }

    Matrix *X_T = matrix_transpose(X);
    Matrix *dW1 = matrix_multiply_gpu(X_T, dZ1);

    Matrix *db1 = matrix_create(1, net->b1->cols);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < dZ1->cols; j++) {
            db1->data[j] += dZ1->data[i * dZ1->cols + j];
        }
    }

    float scalar = lr / m;

    for (int i = 0; i < net->W1->rows * net->W1->cols; i++) {
        net->W1->data[i] -= scalar * dW1->data[i];
    }
    for (int i = 0; i < net->b1->cols; i++) {
        net->b1->data[i] -= scalar * db1->data[i];
    }
    for (int i = 0; i < net->W2->rows * net->W2->cols; i++) {
        net->W2->data[i] -= scalar * dW2->data[i];
    }
    for (int i = 0; i < net->b2->cols; i++) {
        net->b2->data[i] -= scalar * db2->data[i];
    }

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

int main() {
    srand(42);

    const char *img_path = "../data/train-images.idx3-ubyte";
    const char *lbl_path = "../data/train-labels.idx1-ubyte";

    // 1. Verificación básica de GPU
    int deviceCount = 0;
    cudaCheckError(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("!!! ERROR: No se detectaron GPUs NVIDIA CUDA !!!\n");
        return 1;
    }

    cudaCheckError(cudaSetDevice(0));

    cudaDeviceProp prop;
    cudaCheckError(cudaGetDeviceProperties(&prop, 0));
    printf("--- INICIANDO CUDA --- \n");
    printf("GPU: %s\n", prop.name);

    FILE *f_img = fopen(img_path, "rb");
    FILE *f_lbl = fopen(lbl_path, "rb");
    if (!f_img || !f_lbl) {
        printf("Error data files\n");
        return 1;
    }

    read_int(f_img);           // magic
    int num_imgs = read_int(f_img);
    int rows = read_int(f_img);
    int cols = read_int(f_img);
    read_int(f_lbl);           // magic
    read_int(f_lbl);           // num labels (igual a num_imgs en MNIST normalmente)

    int LIMIT = 60000;
    if (LIMIT > num_imgs) LIMIT = num_imgs;

    printf("Cargando %d imagenes...\n", LIMIT);

    Matrix *X_train = matrix_create(LIMIT, rows * cols);
    unsigned char *y_temp = (unsigned char*)malloc(LIMIT);

    for (int i = 0; i < LIMIT; i++) {
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel = 0;
            fread(&pixel, 1, 1, f_img);
            X_train->data[i * X_train->cols + j] = (float)pixel / 255.0f;
        }
    }

    fread(y_temp, 1, LIMIT, f_lbl);

    Matrix *Y_train = to_one_hot(y_temp, LIMIT, 10);
    fclose(f_img);
    fclose(f_lbl);

    MLP *mlp = mlp_create(784, 256, 10);
    int EPOCHS = 10;
    int BATCH_SIZE = 64;
    float LR = 0.1f;

    printf("Entrenando %d epocas en GPU...\n", EPOCHS);

    clock_t start = clock();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        int correct = 0;
        for (int i = 0; i < LIMIT; i += BATCH_SIZE) {
            int current_batch = (LIMIT - i < BATCH_SIZE) ? (LIMIT - i) : BATCH_SIZE;
            Matrix *X_batch = matrix_create(current_batch, 784);
            Matrix *Y_batch = matrix_create(current_batch, 10);

            for (int r = 0; r < current_batch; r++) {
                for (int c = 0; c < 784; c++) {
                    X_batch->data[r * 784 + c] = X_train->data[(i + r) * 784 + c];
                }
                for (int c = 0; c < 10; c++) {
                    Y_batch->data[r * 10 + c] = Y_train->data[(i + r) * 10 + c];
                }
            }

            mlp_forward(mlp, X_batch);
            mlp_backward(mlp, X_batch, Y_batch, LR);

            for (int r = 0; r < current_batch; r++) {
                if (get_argmax(mlp->A2, r) == (int)y_temp[i + r]) correct++;
            }

            matrix_free(X_batch);
            matrix_free(Y_batch);
        }
        printf("Epoch %d | Accuracy: %.2f%%\n",
               epoch + 1, (float)correct / LIMIT * 100.0f);
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\n--- RESULTADO CUDA ---\n");
    printf("Tiempo: %.4f segundos\n", time_taken);

    free(y_temp);
    matrix_free(X_train);
    matrix_free(Y_train);

    cudaDeviceReset();
    return 0;
}
