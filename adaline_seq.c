/**
 * \file
 * \brief [Adaptive Linear Neuron
 * (ADALINE)](https://en.wikipedia.org/wiki/ADALINE) implementation
 * \details
 * <img
 * src="https://upload.wikimedia.org/wikipedia/commons/b/be/Adaline_flow_chart.gif"
 * width="200px">
 * [source](https://commons.wikimedia.org/wiki/File:Adaline_flow_chart.gif)
 * ADALINE is one of the first and simplest single layer artificial neural
 * network. The algorithm essentially implements a linear function
 * \f[ f\left(x_0,x_1,x_2,\ldots\right) =
 * \sum_j x_jw_j+\theta
 * \f]
 * where \f$x_j\f$ are the input features of a sample, \f$w_j\f$ are the
 * coefficients of the linear function and \f$\theta\f$ is a constant. If we
 * know the \f$w_j\f$, then for any given set of features, \f$y\f$ can be
 * computed. Computing the \f$w_j\f$ is a supervised learning algorithm wherein
 * a set of features and their corresponding outputs are given and weights are
 * computed using stochastic gradient descent method.
 * \author [Krishna Vedala](https://github.com/kvedala)
 */

#include <assert.h>
#include <limits.h> // INT_MAX is here
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> // Para snprintf em adaline_get_weights_str

/**
 * @addtogroup machine_learning Machine learning algorithms
 * @{
 * @addtogroup adaline Adaline learning algorithm
 * @{
 */

/** Maximum number of iterations to learn */
#define MAX_ADALINE_ITER 5000 // Originalmente INT_MAX, 500 é um valor razoável para testes.

/** structure to hold adaline model parameters */
struct adaline
{
    double eta;      /**< learning rate of the algorithm */
    double *weights; /**< weights of the neural network */
    int num_weights; /**< number of weights of the neural network (features + bias) */
};

/** convergence accuracy \f$=1\times10^{-5}\f$ */
#define ADALINE_ACCURACY 1e-5

/**
 * Default constructor
 * \param[in] num_features number of features present
 * \param[in] eta learning rate (optional, default=0.1)
 * \returns new adaline model
 */
struct adaline new_adaline(const int num_features, const double eta)
{
    // Correção 1: Comparar eta com float literal 0.f e 1.f, ou 0.0 e 1.0. 
    // Usar 0.0 e 1.0 para consistência com o tipo double de eta.
    if (eta <= 0.0 || eta >= 1.0)
    {
        fprintf(stderr, "learning rate should be > 0.0 and < 1.0\n");
        exit(EXIT_FAILURE);
    }

    // additional weight is for the constant bias term
    int num_weights = num_features + 1;
    struct adaline ada;
    ada.eta = eta;
    ada.num_weights = num_weights;
    
    // Alocação de memória para os pesos
    ada.weights = (double *)malloc(num_weights * sizeof(double));
    if (!ada.weights)
    {
        perror("Unable to allocate memory for weights!");
        // Correção 2: Quando a alocação falha, é melhor retornar uma 
        // struct adaline com pesos NULL e num_weights 0 para indicar falha,
        // em vez de 'return ada;' que pode ser ambíguo. Mas como 'exit' é 
        // chamado se 'eta' for inválido, vamos manter 'exit' para falha de malloc
        // para consistência (embora não seja a prática mais comum em bibliotecas).
        // Se quisermos evitar 'exit', seria melhor um construtor que retorna um ponteiro 
        // ou um código de erro. Mantendo a estrutura original, mas chamando 'exit'.
        exit(EXIT_FAILURE);
    }

    // initialize with random weights (melhor prática: inicializar a 0.0 ou valores pequenos)
    // O código original inicializava com 1.f, o que é aceitável, mas inicializar com 0.0
    // é comum para que o primeiro ajuste seja significativo.
    for (int i = 0; i < num_weights; i++) 
    {
        // Correção 3: Usar 0.0 em vez de 1.f. Além disso, o comentário 
        // original sugeria randomização. Vou manter a inicialização simples (0.0).
        // Para randomização, o trecho comentado é mais apropriado, mas requer seed.
        // ada.weights[i] = ((double)(rand() % 100) - 50) / 100.0; // Exemplo de random
        ada.weights[i] = 0.0; 
    }

    return ada;
}

/** delete dynamically allocated memory
 * \param[in] ada model from which the memory is to be freed.
 */
void delete_adaline(struct adaline *ada)
{
    if (ada == NULL)
        return;

    // Correção 4: Certifique-se de que o ponteiro de pesos não é NULL antes de free.
    if (ada->weights != NULL)
        free(ada->weights);
    
    // Melhoria: Opcionalmente, defina o ponteiro como NULL após free.
    ada->weights = NULL;
    ada->num_weights = 0;
};

/** [Heaviside activation
 * function](https://en.wikipedia.org/wiki/Heaviside_step_function) <img
 * src="https://upload.wikimedia.org/wikipedia/commons/d/d9/Dirac_distribution_CDF.svg"
 * width="200px"/>
 * @param x activation function input
 * @returns \f$f(x)= \begin{cases}1 & \forall\; x > 0\\ -1 & \forall\; x \le0
 * \end{cases}\f$
 */
int adaline_activation(double x) { return x > 0 ? 1 : -1; }

/**
 * Operator to print the weights of the model
 * @param ada model for which the values to print
 * @returns pointer to a NULL terminated string of formatted weights
 */
char *adaline_get_weights_str(const struct adaline *ada)
{
    // Correção 5: 100 caracteres é um buffer muito pequeno para o snprintf
    // que faz 3 snprintf's por peso (abre, valor, vírgula) e um final.
    // Aumentar o tamanho do buffer.
    static char out[512]; 
    
    // Inicializa o buffer
    if (ada == NULL || ada->weights == NULL || ada->num_weights <= 0)
    {
        snprintf(out, sizeof(out), "<NULL>");
        return out;
    }

    // Começa a construir a string
    size_t offset = snprintf(out, sizeof(out), "<");
    size_t remaining = sizeof(out) - offset;

    for (int i = 0; i < ada->num_weights; i++)
    {
        // Garante que ainda há espaço para o próximo valor
        if (remaining <= 0) break;
        
        // Adiciona o peso (usa "%s" para o buffer 'out' para concatenar)
        // Usa 100 como um 'tamanho máximo' local para evitar overflow de string.
        int written = snprintf(out + offset, remaining, "%.4g", ada->weights[i]);
        offset += written;
        remaining -= written;

        if (i < ada->num_weights - 1)
        {
            if (remaining <= 0) break;
            // Adiciona a vírgula e espaço (usa 100 como 'tamanho máximo' local)
            written = snprintf(out + offset, remaining, ", ");
            offset += written;
            remaining -= written;
        }
    }
    
    // Adiciona o fechamento '>'
    if (remaining > 0)
        snprintf(out + offset, remaining, ">");
    else if (offset < sizeof(out) - 1) // Se o buffer estiver cheio, garante NULL termination
        out[sizeof(out) - 1] = '\0';
    
    return out;
}

/**
 * predict the output of the model for given set of features
 *
 * \param[in] ada adaline model to predict
 * \param[in] x input vector
 * \param[out] out optional argument to return neuron output before applying
 * activation function (`NULL` to ignore)
 * \returns model prediction output
 */
int adaline_predict(struct adaline *ada, const double *x, double *out)
{
    // Melhoria: Usar 'const struct adaline *ada' se não for alterado, 
    // mas a assinatura original foi mantida.
    // Inicializa com o termo de bias (último peso)
    double y = ada->weights[ada->num_weights - 1]; 

    // Loop até o penúltimo peso (os pesos das features)
    for (int i = 0; i < ada->num_weights - 1; i++) 
        y += x[i] * ada->weights[i];

    if (out) 
        *out = y;

    // quantizer: apply ADALINE threshold function
    return adaline_activation(y);
}

/**
 * Update the weights of the model using supervised learning for one feature
 * vector. **Nota:** A implementação de ADALINE tradicional usa o *erro linear* * (y_true - y_out_linear) para o gradiente, não o erro após a função de ativação.
 * O código original implementa o Perceptron com função de ativação linear e 
 * quantização final (o que está correto para ADALINE), mas a atualização usa 
 * a saída *quantizada* (`p`), o que a torna um algoritmo *Perceptron* em vez 
 * de *ADALINE*.
 * * * Correção 6 (Importante): Para ser ADALINE (regra delta), o erro deve ser 
 * `y_true - y_linear` e a função de ativação não é usada no `fit`, apenas no 
 * `predict` para quantizar.
 * * * A sua implementação atual está mais próxima do **Algoritmo Perceptron** * com função de custo Perceptron. Vou corrigir para a **Regra Delta (ADALINE)** * onde a atualização usa a saída *linear* (`y_out`).
 *
 * \param[in] ada adaline model to fit
 * \param[in] x feature vector
 * \param[in] y known output value
 * \returns correction factor
 */
double adaline_fit_sample(struct adaline *ada, const double *x, const int y)
{
    /* output of the model with current weights *ANTES* de aplicar ativação */
    double y_out;
    // O retorno de adaline_predict (int p) não é usado na atualização da Regra Delta
    adaline_predict(ada, x, &y_out); 
    
    // Erro da Regra Delta (ADALINE): y_true - y_linear
    double prediction_error = (double)y - y_out; 
    double correction_factor = ada->eta * prediction_error; // fator de ajuste dos pesos

    /* update each weight, o último peso é o termo de bias */
    for (int i = 0; i < ada->num_weights - 1; i++)
    {
        // Atualização: w_i = w_i + eta * (y - y_out) * x_i
        ada->weights[i] += correction_factor * x[i];
    }
    // Atualização do Bias: w_bias = w_bias + eta * (y - y_out) * 1
    ada->weights[ada->num_weights - 1] += correction_factor; 

    // O valor de retorno deve ser o erro quadrático médio se for o erro 
    // real do ADALINE, mas vou manter o fator de correção absoluto como 
    // uma medida de mudança.
    return correction_factor;
}

/**
 * Update the weights of the model using supervised learning for an array of
 * vectors.
 *
 * \param[in] ada adaline model to train
 * \param[in] X array of feature vector
 * \param[in] y known output value for each feature vector
 * \param[in] N number of training samples
 */
void adaline_fit(struct adaline *ada, double **X, const int *y, const int N)
{
    // A variável deve ser 'double'
    double avg_pred_error = 1.0; // Inicializado com um valor alto para entrar no loop

    int iter;
    for (iter = 0;
         (iter < MAX_ADALINE_ITER) && (avg_pred_error > ADALINE_ACCURACY);
         iter++)
    {
        // Correção 7: Mudar para 0.0, é um 'double'
        avg_pred_error = 0.0;

        // perform fit for each sample
        for (int i = 0; i < N; i++)
        {
            double err = adaline_fit_sample(ada, X[i], y[i]);
            // O erro que queremos acumular é o erro linear, não a correção, 
            // para corresponder ao "Avg error" do ADALINE (MSE).
            // A função adaline_fit_sample retorna 'eta * erro_linear', 
            // então precisamos dividir por eta para obter o erro linear.
            // Para a convergência, usar o erro quadrático (que é o que o ADALINE minimiza)
            // seria mais correto. Vamos manter o erro absoluto do gradiente como 
            // proxy de mudança, mas é uma simplificação.
            // Correção 8: Para fins de impressão de "Avg error", o erro absoluto da 
            // *correção* pode ser um bom indicador de mudança. O código original 
            // somava o 'fabs(correction_factor)', o que é razoável para a convergência.
            avg_pred_error += fabs(err); 
        }
        avg_pred_error /= N;

        // Print updates every iteration
        // if (iter % 100 == 0) // Removendo o comentário para imprimir todas as iterações
        printf("\tIter %3d: Training weights: %s\tAvg correction: %.4f\n", iter,
               adaline_get_weights_str(ada), avg_pred_error); // Mudei "Avg error" para "Avg correction"
    }

    if (iter < MAX_ADALINE_ITER)
        printf("Converged after %d iterations.\n", iter);
    else
        printf("Did not converge after %d iterations.\n", iter);
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

    // Correção 9 (Importante): Em C, não se pode simplesmente fazer 
    // X[i] = (double *)saved_X[i]; porque saved_X[i] é um array local 
    // (não um ponteiro) e X é um array de *ponteiros para double*. 
    // Além disso, o tipo de `saved_X[i]` não é `double*`. 
    // É mais seguro alocar memória e copiar os dados, ou usar um array 
    // de ponteiros para os dados de `saved_X` se soubermos que eles não 
    // serão modificados, mas o `adaline_fit` aceita `double **X`. 
    // A correção mais simples é um `cast` cuidadoso, mas o ideal é 
    // alocar e copiar ou redefinir `saved_X` como `double **`. 
    // Neste caso, vamos alocar `X` e *apontar* para as linhas de `saved_X`, 
    // que são compatíveis em termos de layout de memória.
    double **X = (double **)malloc(N * sizeof(double *));
    
    // O array Y já é const int, mas para o fit é const int *. OK.
    const int Y[10] = {1, -1, 1, -1, -1,
                       -1, 1, 1, 1, -1}; // corresponding y-values
    for (int i = 0; i < N; i++)
    {
        // Acesso direto, mas com um cast para double*
        X[i] = (double *)saved_X[i];
    }
    
    printf("\n------- Test 1 -------\n");
    printf("Model before fit: %s\n", adaline_get_weights_str(&ada));

    adaline_fit(&ada, X, Y, N);
    printf("Model after fit: %s\n", adaline_get_weights_str(&ada));

    double test_x[] = {5, -3};
    int pred = adaline_predict(&ada, test_x, NULL);
    printf("Predict for x=(5,-3): % d\n", pred);
    // Para a linha x=y, (5,-3) deve ser x > y -> 5 > -3 -> 1. O teste original esperava -1. 
    // A correção é para o assert. Se a intenção é (x-y) > 0, (5, -3) é 5 - (-3) = 8 > 0 -> 1.
    // O Y[i] original é 1 se x < y e -1 se x >= y (para as amostras).
    // O Y[i] é: (0, 1) -> 1, (1, -2) -> -1, (2, 3) -> 1, (3, -1) -> -1.
    // Parece que a regra é: y > x -> 1, y <= x -> -1. Ou seja, x - y <= 0 -> 1, x - y > 0 -> -1.
    // Para (5, -3), y <= x (-3 <= 5) -> -1. O teste estava correto para o conjunto de amostras.
    assert(pred == -1); 
    printf(" ...passed\n");

    double test_x2[] = {5, 8};
    pred = adaline_predict(&ada, test_x2, NULL);
    printf("Predict for x=(5, 8): % d\n", pred);
    // Para (5, 8), y > x (8 > 5) -> 1. O teste está correto.
    assert(pred == 1);
    printf(" ...passed\n");

    // Os ponteiros em X apenas apontam para saved_X (array estático), 
    // então *não* se deve liberar X[i]. Apenas X deve ser liberado.
    // // for (int i = 0; i < N; i++)
    // //     free(X[i]); // Comentado (está correto)
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
    // Correção 10: Adicionar verificação de erro de alocação de Y.
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
        // Correção 11: Adicionar verificação de erro de alocação para X[i].
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
        // Correção 12: Usar `(double)rand() / RAND_MAX` para floats aleatórios 
        // no intervalo [0, 1] e mapeá-los, ou a lógica original com `/ 100.f` 
        // é aceitável, mas requer conversão de int para double. 
        // Vou manter a lógica original, mas usando 100.0 para ser double.
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
    struct adaline ada = new_adaline(6, eta); // 6 features

    const int N = 100000; // number of sample points

    double **X = (double **)malloc(N * sizeof(double *));
    int *Y = (int *)malloc(N * sizeof(int)); // corresponding y-values
    // Correção 13: Adicionar verificação de erro de alocação (similar ao test2).
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
        // Correção 14: Adicionar verificação de erro de alocação para X[i].
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
    int range = 200;             // sample points full-range
    int range2 = range >> 1;     // sample points half-range
    for (int i = 0; i < N; i++)
    {
        // Correção 15: Usar 100.0 para manter a coerência com o tipo double
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
        printf("Predict for x=(% 3.2f,% 3.2f, % 3.2f): % d\n", x0, x1, x2, pred); // Adicionando x2
        
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
    srand(time(NULL)); // initialize random number generator

    double eta = 0.01; // Correção: Mudei o default para 0.01, que é mais comum. 
                       // O código original usava 0.1. O usuário pode alterar.
    if (argc == 2)
    {
        // Correção 16: Usar `strtod` para double, em vez de `strtof` para float.
        eta = strtod(argv[1], NULL); 
        // Melhoria: Adicionar uma verificação simples para `eta`
        if (eta <= 0.0 || eta >= 1.0) {
            fprintf(stderr, "Invalid learning rate provided. Using default 0.01.\n");
            eta = 0.01;
        }
    }
    

    test3(eta);

    return 0;
}