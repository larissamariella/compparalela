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

// --- Construtor e Destrutor (Inalterados, exceto limpeza) ---
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

/**
 * Predição na GPU (Opcional, mas recomendada se for fazer muitas predições em lote)
 * Aqui mantive simples para um único vetor, rodando no Host ou Device conforme necessidade.
 */
int adaline_predict(struct adaline *ada, const double *x, double *out)
{
    double y = ada->weights[ada->num_weights - 1]; // Bias (último peso)
    
    // Cálculo simples, geralmente não vale a pena offload para GPU para 1 única amostra
    // a menos que esteja dentro de um loop maior mapeado.
    for (int i = 0; i < ada->num_weights - 1; i++) 
        y += x[i] * ada->weights[i];

    if (out) *out = y;
    return adaline_activation(y);
}

/**
 * Treinamento Otimizado para GPU
 * \param X_flat Array 1D contendo as features contíguas. 
 * Acesso: X[amostra * n_features + feature]
 */
void adaline_fit(struct adaline *ada, const double *X_flat, const int *y, const int N)
{
    // CORREÇÃO 1: Verificação de segurança
    if (ada->num_weights > MAX_GPU_WEIGHTS) {
        fprintf(stderr, "Erro: Numero de pesos (%d) excede o maximo suportado pela GPU (%d).\n", 
                ada->num_weights, MAX_GPU_WEIGHTS);
        exit(EXIT_FAILURE);
    }

    double avg_pred_error = 1.0; 
    int iter;
    int num_weights = ada->num_weights;
    int num_features = num_weights - 1;
    double *weights = ada->weights;
    double eta = ada->eta;

    // CORREÇÃO 2: Alocamos o tamanho MÁXIMO (128) para garantir que a redução
    // não acesse memória inválida, mesmo que usemos apenas 7.
    double *grad_sum = (double *)calloc(MAX_GPU_WEIGHTS, sizeof(double));
    if (!grad_sum) exit(EXIT_FAILURE);

    // Mapeamento: note que mapeamos grad_sum com o tamanho MAX_GPU_WEIGHTS
    #pragma omp target data map(to: X_flat[0:N*num_features], y[0:N]) \
                            map(tofrom: weights[0:num_weights]) \
                            map(alloc: grad_sum[0:MAX_GPU_WEIGHTS])
    {
        for (iter = 0; (iter < MAX_ADALINE_ITER) && (avg_pred_error > ADALINE_ACCURACY); iter++)
        {
            avg_pred_error = 0.0;
            
            // Zera o buffer (usando o tamanho fixo para garantir limpeza total)
            #pragma omp target teams distribute parallel for
            for(int j=0; j<MAX_GPU_WEIGHTS; j++) {
                grad_sum[j] = 0.0;
            }

            // CORREÇÃO 3: Usamos a constante [:MAX_GPU_WEIGHTS] na cláusula reduction
            #pragma omp target teams distribute parallel for \
                reduction(+:avg_pred_error) \
                reduction(+:grad_sum[:MAX_GPU_WEIGHTS])
            for (int i = 0; i < N; i++)
            {
                // O código interno continua usando 'num_weights' e 'num_features' reais
                // para não desperdiçar processamento matemático.
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

            // Atualização dos pesos (apenas os necessários)
            #pragma omp target teams distribute parallel for
            for (int j = 0; j < num_weights; j++) {
                weights[j] += eta * (grad_sum[j] / N);
            }
            #pragma omp target update from(weights[0:num_weights])

            // 4. Imprime igual ao código original
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

/**
 * Teste pesado (Test 3) adaptado para memória linear
 */
void test3(double eta)
{
    // Aumentei N para justificar o uso de GPU (1 milhão de pontos)
    const int N = 5000; 
    const int n_features_input = 6; 
    
    struct adaline ada = new_adaline(n_features_input, eta);

    // Alocação Linear (Crucial para GPU)
    double *X = (double *)malloc(N * n_features_input * sizeof(double));
    int *Y = (int *)malloc(N * sizeof(int));

    if (!X || !Y) {
        perror("Memory allocation failed");
        exit(1);
    }

    printf("Generating %d samples...\n", N);
    
    int range = 200;
    int range2 = range >> 1;
    
    // Preenchimento dos dados (na CPU)
    for (int i = 0; i < N; i++)
    {
        double x0 = ((rand() % range) - range2) / 100.0;
        double x1 = ((rand() % range) - range2) / 100.0;
        double x2 = ((rand() % range) - range2) / 100.0;
        
        // Mapeamento linear: indice = i * n_colunas + coluna
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

    // Testes de verificação
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
    
    // Verificando dispositivos disponíveis
    int num_devices = omp_get_num_devices();
    printf("Number of GPU devices available: %d\n", num_devices);
    
    test3(eta);
    return 0;
}