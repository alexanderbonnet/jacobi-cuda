#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define N (2048 * 2048)
#define THREADS_PER_BLOCK 512

double *init_x(double value) {
    double *x_vect = (double *)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        x_vect[i] = value;
    }
    return x_vect;
}

__global__ void dot(double *a, double *b, double *c) {
    __shared__ double temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = a[index] * b[index];

    __syncthreads();

    if (0 == threadIdx.x) {
        double sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++) sum += temp[i];
        atomicAdd(c, sum);
    }
}

__global__ void squared_norm_of_diff(double *a, double *b, double *c) {
    __shared__ double temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    double diff = a[index] - b[index];
    temp[threadIdx.x] = diff * diff;

    __syncthreads();

    if (0 == threadIdx.x) {
        double sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++) sum += temp[i];
        atomicAdd(c, sum);
    }
}

int main(void) {
    double *a, *b, *c;
    double *dev_a, *dev_b, *dev_c;
    double size = N * sizeof(double);

    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, sizeof(double));

    a = (double *)malloc(size);
    b = (double *)malloc(size);
    c = (double *)malloc(sizeof(double));

    a = init_x(1.0);
    b = init_x(2.0);

    // copy inputs to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // launch dot() kernel with 1 block and N threads
    squared_norm_of_diff<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        dev_a, dev_b, dev_c);

    // copy device result back to host copy of c
    cudaMemcpy(c, dev_c, sizeof(double), cudaMemcpyDeviceToHost);

    printf("%f\n", *c);

    free(a);
    free(b);
    free(c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}