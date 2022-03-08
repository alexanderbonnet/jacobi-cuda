#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define N 512

double* init_x(double value) {
    double* x_vect = malloc(sizeof(double) * N);
    for (int i = 0; i < size; i++) {
        x_vect[i] = value;
    }
    return x_vect;
}

__global__ squared_norm_of_diff(double* a_vec, double* b_vec, double* c,
                                int size) {
    __shared__ int temp[size];
    double diff = a[threadIdx.x] - b[threadIdx.x];
    temp[threadIdx.x] = diff * diff;

    __syncthreads();

    if (0 == threadIdx.x) {
        double sum = 0;
        for (int i = 0; i < size; i++) {
            sum += temp[i];
        }
        *c = sum;
    }
}

__global__ void dot(double* a, double* b, double* c) {
    __shared__ int temp[size];
    temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
    __syncthreads();
    if (0 == threadIdx.x) {
        double sum = 0;
        for (int i = 0; i < size; i++) {
            sum += temp[i];
        }
        *c = sum;
    }
}

int main(void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int size = N * sizeof(double);

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, sizeof(double));

    a = (double*)malloc(size);
    b = (double*)malloc(size);
    c = (double*)malloc(sizeof(double));

    a = init_x(1.0);
    b = init_x(2.0);

    // copy inputs to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // launch dot() kernel with 1 block and N threads
    dot<<<1, N>>>(dev_a, dev_b, dev_c);

    // copy device result back to host copy of c
    cudaMemcpy(c, dev_c, sizeof(double), cudaMemcpyDeviceToHost);

    free(a);
    free(b);
    free(c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}