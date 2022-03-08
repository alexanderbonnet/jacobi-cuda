#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ increment_x_inplace(double* x_vec, double* x_tmp, double* a_mat,
                               double* b_vec {
    double sum = 0;
    int idx = threadIdx.x;
    for (int j = 0; j < N, j++) {
        if (i != j) {
            sum += a_mat[idx * N + j] * x_tmp[j];
        }
    }
    x_vec[idx] = (b_vec[idx] - sum) / a_mat[idx * N + idx];
}

__global__ squared_norm_of_diff(double* a_vec, double* b_vec) {}

__global__ void dot(double* a, double* b, double* c) {
    __shared__ int temp[N];
    temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
    __syncthreads();
    if (0 == threadIdx.x) {
        double sum = 0;
        for (int i = 0; i < N; i++) {
            sum += temp[i];
        }
        *c = sum;
    }
}
