/**
 * \file
 * \brief [Adaptive Linear Neuron (ADALINE)](https://en.wikipedia.org/wiki/ADALINE)
 * implementation using CUDA for GPU acceleration
 * \details
 * This is a GPU-accelerated version of the ADALINE algorithm using CUDA.
 * The algorithm implements a linear function:
 * \f[ f\left(x_0,x_1,x_2,\ldots\right) = \sum_j x_jw_j+\theta \f]
 * 
 * Key GPU optimizations:
 * - Parallel forward pass using CUDA kernels
 * - Parallel gradient computation
 * - Efficient memory transfers between CPU and GPU
 * - Atomic operations for reduction
 * 
 * \author [GPU Implementation]
 */

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

/**
 * @addtogroup machine_learning Machine learning algorithms
 * @{
 * @addtogroup adaline_cuda Adaline CUDA learning algorithm
 * @{
 */

/** Maximum number of iterations to learn */
#define MAX_ADALINE_ITER 5000

/** Threads per block for GPU kernels */
#define THREADS_PER_BLOCK 256

/** structure to hold adaline model parameters */
struct adaline
{
    double eta;          /**< learning rate of the algorithm */
    double *weights;     /**< weights on CPU */
    double *d_weights;   /**< weights on GPU device */
    int num_weights;     /**< number of weights (features + bias) */
};

/** convergence accuracy */
#define ADALINE_ACCURACY 1e-5

/**
 * CUDA error checking macro
 */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = call;                                          \
        if (error != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(error));                 \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while(0)

/**
 * Default constructor - Initialize ADALINE model
 * \param[in] num_features number of features present
 * \param[in] eta learning rate (optional, default=0.1)
 * \returns new adaline model
 */
struct adaline new_adaline(const int num_features, const double eta)
{
    if (eta <= 0.0 || eta >= 1.0)
    {
        fprintf(stderr, "learning rate should be > 0.0 and < 1.0\n");
        exit(EXIT_FAILURE);
    }

    int num_weights = num_features + 1;
    struct adaline ada;
    ada.eta = eta;
    ada.num_weights = num_weights;
    
    // Allocate memory on CPU
    ada.weights = (double *)malloc(num_weights * sizeof(double));
    if (!ada.weights)
    {
        perror("Unable to allocate memory for weights on CPU!");
        exit(EXIT_FAILURE);
    }

    // Allocate memory on GPU device
    CUDA_CHECK(cudaMalloc((void **)&ada.d_weights, num_weights * sizeof(double)));

    // Initialize weights to 0
    for (int i = 0; i < num_weights; i++) 
    {
        ada.weights[i] = 0.0; 
    }

    // Copy initial weights to GPU
    CUDA_CHECK(cudaMemcpy(ada.d_weights, ada.weights, 
                          num_weights * sizeof(double), 
                          cudaMemcpyHostToDevice));

    return ada;
}

/**
 * Delete dynamically allocated memory (CPU and GPU)
 * \param[in] ada model from which the memory is to be freed
 */
void delete_adaline(struct adaline *ada)
{
    if (ada == NULL)
        return;

    if (ada->weights != NULL)
        free(ada->weights);
    
    if (ada->d_weights != NULL)
        CUDA_CHECK(cudaFree(ada->d_weights));
    
    ada->weights = NULL;
    ada->d_weights = NULL;
    ada->num_weights = 0;
}

/**
 * Heaviside activation function
 * @param x activation function input
 * @returns 1 if x > 0, -1 otherwise
 */
__device__ int adaline_activation_device(double x) 
{ 
    return x > 0 ? 1 : -1; 
}

int adaline_activation(double x) 
{ 
    return x > 0 ? 1 : -1; 
}

/**
 * GPU kernel: Compute linear output (prediction) for all samples
 * Each thread processes one sample
 * 
 * \param[in] X Input feature matrix (device)
 * \param[in] weights Model weights (device)
 * \param[out] y_out Output values (device)
 * \param[in] N Number of samples
 * \param[in] num_features Number of features
 */
__global__ void kernel_predict(const double *X, const double *weights,
                               double *y_out, const int N, 
                               const int num_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N)
    {
        // Compute linear output: bias + sum(x_i * w_i)
        double y = weights[num_features]; // bias term (last weight)
        
        for (int j = 0; j < num_features; j++)
        {
            y += X[idx * num_features + j] * weights[j];
        }
        
        y_out[idx] = y;
    }
}

/**
 * GPU kernel: Compute gradients and accumulate error
 * Each block computes gradients for its assigned samples
 * 
 * \param[in] X Input feature matrix (device)
 * \param[in] Y True labels (device)
 * \param[in] y_out Predicted outputs (device)
 * \param[out] grad_sum Accumulated gradients (device)
 * \param[out] error_sum Accumulated absolute error (device)
 * \param[in] N Number of samples
 * \param[in] num_weights Number of weights (features + 1)
 * \param[in] eta Learning rate
 */
__global__ void kernel_compute_gradients(const double *X, const int *Y,
                                          const double *y_out,
                                          double *grad_sum, 
                                          double *error_sum,
                                          const int N,
                                          const int num_weights,
                                          const double eta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Shared memory for local reduction
    __shared__ double shared_error[THREADS_PER_BLOCK];
    
    shared_error[threadIdx.x] = 0.0;
    __syncthreads();
    
    // Each thread processes multiple samples (stride loop)
    for (int i = idx; i < N; i += stride)
    {
        // Compute error: y_true - y_linear
        double prediction_error = (double)Y[i] - y_out[i];
        
        // Accumulate absolute error for convergence check
        atomicAdd(error_sum, fabs(eta * prediction_error));
        
        // Update gradients for each weight
        for (int j = 0; j < num_weights - 1; j++)
        {
            atomicAdd(&grad_sum[j], prediction_error * X[i * (num_weights - 1) + j]);
        }
        
        // Update bias gradient
        atomicAdd(&grad_sum[num_weights - 1], prediction_error);
    }
}

/**
 * GPU kernel: Update weights using accumulated gradients
 * 
 * \param[in,out] weights Model weights (device)
 * \param[in] grad_sum Accumulated gradients (device)
 * \param[in] num_weights Number of weights
 * \param[in] eta Learning rate
 * \param[in] N Number of samples
 */
__global__ void kernel_update_weights(double *weights, const double *grad_sum,
                                      const int num_weights,
                                      const double eta, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_weights)
    {
        // Update: w = w + eta * (grad_sum / N)
        weights[idx] += eta * (grad_sum[idx] / N);
    }
}

/**
 * Operator to print the weights of the model
 * @param ada model for which the values to print
 * @returns pointer to a NULL terminated string of formatted weights
 */
char *adaline_get_weights_str(const struct adaline *ada)
{
    static char out[512]; 
    
    if (ada == NULL || ada->weights == NULL || ada->num_weights <= 0)
    {
        snprintf(out, sizeof(out), "<NULL>");
        return out;
    }

    size_t offset = snprintf(out, sizeof(out), "<");
    size_t remaining = sizeof(out) - offset;

    for (int i = 0; i < ada->num_weights; i++)
    {
        if (remaining <= 0) break;
        
        int written = snprintf(out + offset, remaining, "%.4g", ada->weights[i]);
        offset += written;
        remaining -= written;

        if (i < ada->num_weights - 1)
        {
            if (remaining <= 0) break;
            written = snprintf(out + offset, remaining, ", ");
            offset += written;
            remaining -= written;
        }
    }
    
    if (remaining > 0)
        snprintf(out + offset, remaining, ">");
    else if (offset < sizeof(out) - 1)
        out[sizeof(out) - 1] = '\0';
    
    return out;
}

/**
 * Predict the output of the model for given set of features
 *
 * \param[in] ada adaline model to predict
 * \param[in] x input vector (on CPU)
 * \param[out] out optional argument to return neuron output before activation
 * \returns model prediction output
 */
int adaline_predict(struct adaline *ada, const double *x, double *out)
{
    double y = ada->weights[ada->num_weights - 1]; // bias

    for (int i = 0; i < ada->num_weights - 1; i++) 
        y += x[i] * ada->weights[i];

    if (out) 
        *out = y;

    return adaline_activation(y);
}

/**
 * Train the ADALINE model using GPU acceleration
 *
 * \param[in] ada adaline model to train
 * \param[in] X array of feature vectors (on CPU)
 * \param[in] Y known output values (on CPU)
 * \param[in] N number of training samples
 */
void adaline_fit(struct adaline *ada, double **X, const int *Y, const int N)
{
    double avg_pred_error = 1.0;
    int iter;

    // Flatten X array for GPU transfer (N samples × (num_features) values)
    int num_features = ada->num_weights - 1;
    double *X_flat = (double *)malloc(N * num_features * sizeof(double));
    if (!X_flat)
    {
        perror("Unable to allocate memory for flattened X!");
        return;
    }

    // Flatten the 2D array X into 1D
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            X_flat[i * num_features + j] = X[i][j];
        }
    }

    // Allocate GPU memory for data
    double *d_X = NULL;
    int *d_Y = NULL;
    double *d_y_out = NULL;
    double *d_grad_sum = NULL;
    double *d_error_sum = NULL;
    double h_error_sum = 0.0;

    CUDA_CHECK(cudaMalloc((void **)&d_X, N * num_features * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_Y, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_y_out, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_grad_sum, ada->num_weights * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_error_sum, sizeof(double)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_X, X_flat, N * num_features * sizeof(double), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, Y, N * sizeof(int), cudaMemcpyHostToDevice));

    // Calculate grid and block dimensions
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int weight_blocks = (ada->num_weights + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("\n------- ADALINE CUDA Training -------\n");
    printf("GPU Training: %d samples, %d features, %d blocks\n", N, num_features, blocks);
    printf("Model before fit: %s\n", adaline_get_weights_str(ada));

    // Training loop
    for (iter = 0;
         (iter < MAX_ADALINE_ITER) && (avg_pred_error > ADALINE_ACCURACY);
         iter++)
    {
        avg_pred_error = 0.0;
        h_error_sum = 0.0;

        // Reset gradient accumulator on GPU
        CUDA_CHECK(cudaMemset(d_grad_sum, 0, ada->num_weights * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_error_sum, 0, sizeof(double)));

        // Phase 1: Compute predictions
        kernel_predict<<<blocks, THREADS_PER_BLOCK>>>(
            d_X, ada->d_weights, d_y_out, N, num_features);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Phase 2: Compute gradients and accumulate error
        kernel_compute_gradients<<<blocks, THREADS_PER_BLOCK>>>(
            d_X, d_Y, d_y_out, d_grad_sum, d_error_sum, N,
            ada->num_weights, ada->eta);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Phase 3: Update weights
        kernel_update_weights<<<weight_blocks, THREADS_PER_BLOCK>>>(
            ada->d_weights, d_grad_sum, ada->num_weights, ada->eta, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy error sum and weights back to CPU
        CUDA_CHECK(cudaMemcpy(&h_error_sum, d_error_sum, sizeof(double), 
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ada->weights, ada->d_weights, 
                              ada->num_weights * sizeof(double),
                              cudaMemcpyDeviceToHost));

        avg_pred_error = h_error_sum / N;

        if (iter % 10 == 0 || iter < 3)
            printf("\tIter %3d: Training weights: %s\tAvg correction: %.4f\n", 
                   iter, adaline_get_weights_str(ada), avg_pred_error);
    }

    printf("Model after fit: %s\n", adaline_get_weights_str(ada));
    if (iter < MAX_ADALINE_ITER)
        printf("Converged after %d iterations.\n", iter);
    else
        printf("Did not converge after %d iterations.\n", iter);

    // Cleanup GPU memory
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_y_out));
    CUDA_CHECK(cudaFree(d_grad_sum));
    CUDA_CHECK(cudaFree(d_error_sum));
    free(X_flat);
}

/** @}
 * @}
 */

/**
 * Test 1: Predict points in 2D coordinate system above the line x=y
 * \param[in] eta learning rate
 */
void test1(double eta)
{
    struct adaline ada = new_adaline(2, eta);

    const int N = 10;
    const double saved_X[10][2] = {{0, 1},    {1, -2},   {2, 3},   {3, -1},
                                   {4, 1},    {6, -5},   {-7, -3}, {-8, 5},
                                   {-9, 2}, {-10, -15}};

    double **X = (double **)malloc(N * sizeof(double *));
    const int Y[10] = {1, -1, 1, -1, -1, -1, 1, 1, 1, -1};
    
    for (int i = 0; i < N; i++)
    {
        X[i] = (double *)saved_X[i];
    }
    
    printf("\n------- Test 1 -------\n");
    printf("Model before fit: %s\n", adaline_get_weights_str(&ada));

    adaline_fit(&ada, X, Y, N);
    printf("Model after fit: %s\n", adaline_get_weights_str(&ada));

    double test_x[] = {5, -3};
    int pred = adaline_predict(&ada, test_x, NULL);
    printf("Predict for x=(5,-3): %d\n", pred);
    assert(pred == -1);
    printf(" ...passed\n");

    double test_x2[] = {5, 8};
    pred = adaline_predict(&ada, test_x2, NULL);
    printf("Predict for x=(5, 8): %d\n", pred);
    assert(pred == 1);
    printf(" ...passed\n");

    free(X);
    delete_adaline(&ada);
}

/**
 * Test 2: Predict points above the line x+3y=-1
 * \param[in] eta learning rate
 */
void test2(double eta)
{
    struct adaline ada = new_adaline(2, eta);

    const int N = 50;

    double **X = (double **)malloc(N * sizeof(double *));
    int *Y = (int *)malloc(N * sizeof(int));
    
    if (!X || !Y)
    {
        perror("Unable to allocate memory for X or Y!");
        if (X) free(X);
        if (Y) free(Y);
        delete_adaline(&ada);
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < N; i++)
    {
        X[i] = (double *)malloc(2 * sizeof(double));
        if (!X[i])
        {
            perror("Unable to allocate memory for X[i]!");
            for(int j=0; j<i; ++j) free(X[j]);
            free(X);
            free(Y);
            delete_adaline(&ada);
            exit(EXIT_FAILURE);
        }
    }

    int range = 500;
    int range2 = range >> 1;
    
    for (int i = 0; i < N; i++)
    {
        double x0 = ((rand() % range) - range2) / 100.0;
        double x1 = ((rand() % range) - range2) / 100.0;
        X[i][0] = x0;
        X[i][1] = x1;
        Y[i] = (x0 + 3. * x1) > -1 ? 1 : -1;
    }

    printf("\n------- Test 2 -------\n");
    printf("Model before fit: %s\n", adaline_get_weights_str(&ada));

    adaline_fit(&ada, X, Y, N);
    printf("Model after fit: %s\n", adaline_get_weights_str(&ada));

    int N_test_cases = 5;
    double test_x[2];
    
    for (int i = 0; i < N_test_cases; i++)
    {
        double x0 = ((rand() % range) - range2) / 100.0;
        double x1 = ((rand() % range) - range2) / 100.0;
        test_x[0] = x0;
        test_x[1] = x1;
        int pred = adaline_predict(&ada, test_x, NULL);
        printf("Predict for x=(% 3.2f,% 3.2f): %d\n", x0, x1, pred);

        int expected_val = (x0 + 3. * x1) > -1 ? 1 : -1;
        assert(pred == expected_val);
        printf(" ...passed\n");
    }

    for (int i = 0; i < N; i++) free(X[i]);
    free(X);
    free(Y);
    delete_adaline(&ada);
}

/**
 * Test 3: Predict points inside sphere of radius 1
 * \param[in] eta learning rate
 */
void test3(double eta)
{
    struct adaline ada = new_adaline(6, eta);

    const int N = 100000; // Reduced from 1M for reasonable GPU memory usage

    double **X = (double **)malloc(N * sizeof(double *));
    int *Y = (int *)malloc(N * sizeof(int));
    
    if (!X || !Y)
    {
        perror("Unable to allocate memory for X or Y!");
        if (X) free(X);
        if (Y) free(Y);
        delete_adaline(&ada);
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < N; i++)
    {
        X[i] = (double *)malloc(6 * sizeof(double));
        if (!X[i])
        {
            perror("Unable to allocate memory for X[i]!");
            for(int j=0; j<i; ++j) free(X[j]);
            free(X);
            free(Y);
            delete_adaline(&ada);
            exit(EXIT_FAILURE);
        }
    }

    int range = 200;
    int range2 = range >> 1;
    
    for (int i = 0; i < N; i++)
    {
        double x0 = ((rand() % range) - range2) / 100.0;
        double x1 = ((rand() % range) - range2) / 100.0;
        double x2 = ((rand() % range) - range2) / 100.0;
        X[i][0] = x0;
        X[i][1] = x1;
        X[i][2] = x2;
        X[i][3] = x0 * x0;
        X[i][4] = x1 * x1;
        X[i][5] = x2 * x2;
        Y[i] = (x0 * x0 + x1 * x1 + x2 * x2) <= 1 ? 1 : -1;
    }

    printf("\n------- Test 3 -------\n");
    printf("Model before fit: %s\n", adaline_get_weights_str(&ada));

    adaline_fit(&ada, X, Y, N);
    printf("Model after fit: %s\n", adaline_get_weights_str(&ada));

    int N_test_cases = 5;
    double test_x[6];
    
    for (int i = 0; i < N_test_cases; i++)
    {
        double x0 = ((rand() % range) - range2) / 100.0;
        double x1 = ((rand() % range) - range2) / 100.0;
        double x2 = ((rand() % range) - range2) / 100.0;
        test_x[0] = x0;
        test_x[1] = x1;
        test_x[2] = x2;
        test_x[3] = x0 * x0;
        test_x[4] = x1 * x1;
        test_x[5] = x2 * x2;
        int pred = adaline_predict(&ada, test_x, NULL);
        printf("Predict for x=(% 3.2f,% 3.2f, % 3.2f): %d\n", x0, x1, x2, pred);
        
        int expected_val = (x0 * x0 + x1 * x1 + x2 * x2) <= 1 ? 1 : -1;
        assert(pred == expected_val);
        printf(" ...passed\n");
    }

    for (int i = 0; i < N; i++) free(X[i]);
    free(X);
    free(Y);
    delete_adaline(&ada);
}

/** Main function */
int main(int argc, char **argv)
{
    srand(time(NULL));

    double eta = 0.01;
    if (argc == 2)
    {
        eta = strtod(argv[1], NULL);
        if (eta <= 0.0 || eta >= 1.0) {
            fprintf(stderr, "Invalid learning rate provided. Using default 0.01.\n");
            eta = 0.01;
        }
    }

    // Check CUDA capabilities
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0)
    {
        fprintf(stderr, "No CUDA-capable GPU found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    test3(eta);

    return 0;
}
