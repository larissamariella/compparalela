COMANDOS PARA RODAR O CÓDIGO:
adaline_seq.c -> gcc adaline_seq.c -> ./a.out
adaline_paralelo.c -> gcc adaline_paralelo.c -fopenmp -lm
adaline_paralelo_gpu.c -> clang-14 -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_80 -Wno-unknown-cuda-version adaline_paralelo_gpu.c -o adaline_omp_teste -lm -> ./adaline_omp_teste
adaline_paralelo.cu -> nvcc adaline_paralelo.cu -> ./a.out

Nesse trabalho aproveitamos um código de implementação de um modelo de machine learning ADALINE, disponível em https://github.com/TheAlgorithms/C/blob/master/machine_learning/adaline_learning.c
Nos códigos paralelos foram feitas mudanças no código original para aproveitar o máximo possível do paralelismo, como o cálculo do gradiente.


TEMPOS DE EXECUÇÃO: 1000000 SAMPLES, 5000 ÉPOCAS
SEQUENCIAL:
real    3m59.131s
user    3m58.686s
sys     0m0.231s

PARALELO(CPU):
real    0m46.926s
user    3m0.910s
sys     0m0.245s


PARALELO(GPU - OPENMP)
real    0m24.348s
user    7m59.373s
sys     0m6.511s

PARALELO(GPU - CUDA)
real    0m42.272s
user    0m39.873s
sys     0m0.174s