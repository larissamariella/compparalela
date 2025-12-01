/**
 * \file
 * \brief Implementação de ADALINE com Offloading para GPU via OpenMP
 */

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#define MAX_GPU_WEIGHTS 128
#define MAX_ADALINE_ITER 5000 
#define ADALINE_ACCURACY 1e-5

struct adaline
{
    double eta;          
    double *weights;     
    int num_weights;     
};

struct adaline new_adaline(const int num_features, const double eta)
{
    if (eta <= 0.0 || eta >= 1.0) {
        fprintf(stderr, "learning rate should be > 0.0 and < 1.0\n");
        exit(EXIT_FAILURE);
    }
    int num_weights = num_features + 1; // +1 para o Bias
    struct adaline ada;
    ada.eta = eta;
    ada.num_weights = num_weights;
    
    ada.weights = (double *)malloc(num_weights * sizeof(double));
    if (!ada.weights) exit(EXIT_FAILURE);

    for (int i = 0; i < num_weights; i++) ada.weights[i] = 0.0; 
    return ada;
}

void delete_adaline(struct adaline *ada)
{
    if (ada && ada->weights) {
        free(ada->weights);
        ada->weights = NULL;
    }
}

int adaline_activation(double x) { return x > 0 ? 1 : -1; }

char *adaline_get_weights_str(const struct adaline *ada)
{
    static char out[512]; 
    if (!ada || !ada->weights) return "NULL";
    
    size_t offset = snprintf(out, sizeof(out), "<");
    for (int i = 0; i < ada->num_weights; i++) {
        offset += snprintf(out + offset, sizeof(out) - offset, "%.4g%s", 
                           ada->weights[i], (i < ada->num_weights - 1) ? ", " : "");
    }
    snprintf(out + offset, sizeof(out) - offset, ">");
    return out;
}

int adaline_predict(struct adaline *ada, const double *x, double *out)
{
    double y = ada->weights[ada->num_weights - 1]; // Bias (último peso)
    
    for (int i = 0; i < ada->num_weights - 1; i++) 
        y += x[i] * ada->weights[i];

    if (out) *out = y;
    return adaline_activation(y);
}

void adaline_fit(struct adaline *ada, const double *X_flat, const int *y, const int N)
{

    double avg_pred_error = 1.0; 
    int iter;
    int num_weights = ada->num_weights;
    int num_features = num_weights - 1;
    double *weights = ada->weights;
    double eta = ada->eta;

    double *grad_sum = (double *)calloc(MAX_GPU_WEIGHTS, sizeof(double));
    if (!grad_sum) exit(EXIT_FAILURE);

    #pragma omp target data map(to: X_flat[0:N*num_features], y[0:N]) \
                            map(tofrom: weights[0:num_weights]) \
                            map(alloc: grad_sum[0:MAX_GPU_WEIGHTS])
    {
        for (iter = 0; (iter < MAX_ADALINE_ITER) && (avg_pred_error > ADALINE_ACCURACY); iter++)
        {
            avg_pred_error = 0.0;
            
            #pragma omp target teams distribute parallel for
            for(int j=0; j<MAX_GPU_WEIGHTS; j++) {
                grad_sum[j] = 0.0;
            }

            #pragma omp target teams distribute parallel for \
                reduction(+:avg_pred_error) \
                reduction(+:grad_sum[:MAX_GPU_WEIGHTS])
            for (int i = 0; i < N; i++)
            {
                double y_out = weights[num_features]; // Bias
                
                for (int j = 0; j < num_features; j++) {
                    y_out += X_flat[i * num_features + j] * weights[j];
                }

                double prediction_error = (double)y[i] - y_out;
                avg_pred_error += fabs(eta * prediction_error);

                for (int j = 0; j < num_features; j++) {
                    grad_sum[j] += prediction_error * X_flat[i * num_features + j];
                }
                grad_sum[num_features] += prediction_error;
            }

            avg_pred_error /= N;

            #pragma omp target teams distribute parallel for
            for (int j = 0; j < num_weights; j++) {
                weights[j] += eta * (grad_sum[j] / N);
            }
            #pragma omp target update from(weights[0:num_weights])

            printf("\tIter %3d: Training weights: %s\tAvg correction: %.4f\n", 
                   iter, adaline_get_weights_str(ada), avg_pred_error);
        }
    } 

    free(grad_sum);
    
    if (iter < MAX_ADALINE_ITER)
        printf("Converged after %d iterations. Final Error: %.4f\n", iter, avg_pred_error);
    else
        printf("Did not converge. Final Error: %.4f\n", avg_pred_error);
}

/** @}
 * @}
 */

/**
 * test function to predict points in a 2D coordinate system above the line
 * \f$x=y\f$ as +1 and others as -1.
 * Note that each point is defined by 2 values or 2 features.
 * \param[in] eta learning rate (optional, default=0.01)
 */
void test1(double eta)
{
    struct adaline ada = new_adaline(2, eta); // 2 features

    const int N = 10; // number of sample points
    const double saved_X[10][2] = {{0, 1},    {1, -2},   {2, 3},   {3, -1},
                                   {4, 1},    {6, -5},   {-7, -3}, {-8, 5},
                                   {-9, 2}, {-10, -15}};

    double **X = (double **)malloc(N * sizeof(double *));

    const int Y[10] = {1, -1, 1, -1, -1,
                       -1, 1, 1, 1, -1}; // corresponding y-values
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
    printf("Predict for x=(5,-3): % d\n", pred);
    assert(pred == -1); 
    printf(" ...passed\n");

    double test_x2[] = {5, 8};
    pred = adaline_predict(&ada, test_x2, NULL);
    printf("Predict for x=(5, 8): % d\n", pred);
    assert(pred == 1);
    printf(" ...passed\n");

    free(X);
    delete_adaline(&ada);
}

/**
 * test function to predict points in a 2D coordinate system above the line
 * \f$x+3y=-1\f$ as +1 and others as -1.
 * Note that each point is defined by 2 values or 2 features.
 * The function will create random sample points for training and test purposes.
 * \param[in] eta learning rate (optional, default=0.01)
 */
void test2(double eta)
{
    struct adaline ada = new_adaline(2, eta); // 2 features

    const int N = 50; // number of sample points

    double **X = (double **)malloc(N * sizeof(double *));
    int *Y = (int *)malloc(N * sizeof(int)); // corresponding y-values
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

    // generate sample points in the interval
    // [-range2/100 , (range2-1)/100]
    int range = 500;             // sample points full-range
    int range2 = range >> 1;     // sample points half-range
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
        printf("Predict for x=(% 3.2f,% 3.2f): % d\n", x0, x1, pred);

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
 * test function to predict points in a 3D coordinate system lying within the
 * sphere of radius 1 and centre at origin as +1 and others as -1. Note that
 * each point is defined by 3 values but we use 6 features. The function will
 * create random sample points for training and test purposes.
 * The sphere centred at origin and radius 1 is defined as:
 * \f$x^2+y^2+z^2=r^2=1\f$ and if the \f$r^2<1\f$, point lies within the sphere
 * else, outside.
 *
 * \param[in] eta learning rate (optional, default=0.01)
 */
void test3(double eta)
{
    const int N = 100000; 
    const int n_features_input = 6; 
    
    struct adaline ada = new_adaline(n_features_input, eta);

    double *X = (double *)malloc(N * n_features_input * sizeof(double));
    int *Y = (int *)malloc(N * sizeof(int));

    if (!X || !Y) {
        perror("Memory allocation failed");
        exit(1);
    }

    printf("Generating %d samples...\n", N);
    
    int range = 200;
    int range2 = range >> 1;
    
    for (int i = 0; i < N; i++)
    {
        double x0 = ((rand() % range) - range2) / 100.0;
        double x1 = ((rand() % range) - range2) / 100.0;
        double x2 = ((rand() % range) - range2) / 100.0;
        
        size_t idx = i * n_features_input;
        X[idx + 0] = x0;
        X[idx + 1] = x1;
        X[idx + 2] = x2;
        X[idx + 3] = x0 * x0;
        X[idx + 4] = x1 * x1;
        X[idx + 5] = x2 * x2;
        
        Y[i] = (x0 * x0 + x1 * x1 + x2 * x2) <= 1 ? 1 : -1;
    }

    printf("\n------- Test 3 (GPU Accelerated) -------\n");
    printf("Model before fit: %s\n", adaline_get_weights_str(&ada));

    double start = omp_get_wtime();
    adaline_fit(&ada, X, Y, N);
    double end = omp_get_wtime();
    
    printf("Model after fit: %s\n", adaline_get_weights_str(&ada));
    printf("Time elapsed: %.4f seconds\n", end - start);

    double test_x[6] = {0.5, 0.5, 0.5, 0.25, 0.25, 0.25}; // Dentro da esfera
    int pred = adaline_predict(&ada, test_x, NULL);
    printf("Check point (0.5, 0.5, 0.5): Predicted %d (Expected 1)\n", pred);

    free(X);
    free(Y);
    delete_adaline(&ada);
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    double eta = 0.01;
    if (argc == 2) eta = strtod(argv[1], NULL);
    
    int num_devices = omp_get_num_devices();
    printf("Number of GPU devices available: %d\n", num_devices);
    
    test3(eta);
    return 0;
}