#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 2048
#define THREADS_PER_BLOCK 512

double mean(double *x_array, int size) {
    double mean = 0;
    for (int i = 0; i < size; i++) {
        mean += x_array[i];
    }
    mean = mean / size;
    return mean;
}

double *init_x(int size, double value) {
    double *x_vect = (double *)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) {
        x_vect[i] = value;
    }
    return x_vect;
}

double *init_a(int size) {
    double *a_matrix = (double *)malloc(sizeof(double) * size * size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (j == i) {
                a_matrix[i + j * size] = 2 * size + 1;
            } else {
                a_matrix[i + j * size] = 1;
            }
        }
    }
    return a_matrix;
}

void swap(double *&a, double *&b) {
    double *temp = a;
    a = b;
    b = temp;
}

__global__ void criterion(double *a, double *b, double *c) {
    __shared__ double temp[THREADS_PER_BLOCK];

    *c = 0;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    double diff = a[index] - b[index];
    temp[threadIdx.x] = diff * diff;

    __syncthreads();

    if (0 == threadIdx.x) {
        double sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++) {
            sum += temp[i];
        };
        atomicAdd(c, sum);
    }
}

__global__ void increment_x(double *x_new, double *x_old, double *a_mat,
                            double *b_vec) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N) {
        double sum = 0;
        for (int j = 0; j < N; j++) {
            if (index != j) {
                sum += a_mat[index * N + j] * x_old[j];
            }
        }
        x_new[index] = (b_vec[index] - sum) / a_mat[index * N + index];
    }
}

int solve_with_jacobi(double *x_init, double *a_mat, double *b_vec,
                      double epsilon) {
    int nit = 0;
    double eps_2 = epsilon * epsilon;
    double crit = eps_2 + 1;

    double *dev_a, *dev_b, *dev_x_old, *dev_x_new, *dev_crit;

    cudaMalloc((void **)&dev_a, N * N * sizeof(double));
    cudaMalloc((void **)&dev_b, N * sizeof(double));
    cudaMalloc((void **)&dev_x_old, N * sizeof(double));
    cudaMalloc((void **)&dev_x_new, N * sizeof(double));
    cudaMalloc((void **)&dev_crit, sizeof(double));

    // copy inputs to device
    cudaMemcpy(dev_a, a_mat, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b_vec, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x_old, x_init, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x_new, x_init, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_crit, &crit, sizeof(double), cudaMemcpyHostToDevice);

    while (crit > eps_2) {
        increment_x<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            dev_x_new, dev_x_old, dev_a, dev_b);
        criterion<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            dev_x_new, dev_x_old, dev_crit);

        swap(dev_x_old, dev_x_new);
        cudaMemcpy(&crit, dev_crit, sizeof(double), cudaMemcpyDeviceToHost);
        nit += 1;
    }

    cudaMemcpy(x_init, dev_x_new, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_x_old);
    cudaFree(dev_x_new);
    cudaFree(dev_crit);

    return nit;
}

int main(int argc, char *argv[]) {
    int num_executions = 20;

    int nit = 0;
    double result = 0;

    double eps = 1e-6;

    double *a_mat = init_a(N);
    double *x_init = init_x(N, 1);
    double *b_vec = init_x(N, 6);

    double *execution_times = (double *)malloc(sizeof(double) * num_executions);

    struct timeval t1, t2;
    double time = 0;

    printf(
        "running iterative Jacobi algorithm with size = "
        "%d\n",
        N);

    for (int i = 0; i < num_executions; i++) {
        double *x_solve = (double *)malloc(N * sizeof(double));
        memcpy(x_solve, x_init, N * sizeof(double));

        gettimeofday(&t1, 0);

        nit = solve_with_jacobi(x_solve, a_mat, b_vec, eps);

        cudaDeviceSynchronize();
        gettimeofday(&t2, 0);

        time = 1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec;
        time = time / 1000.0;

        execution_times[i] = time;
        printf("exec time = %f \n", time);

        result = x_solve[0];

        free(x_solve);
    }

    double avg_time = mean(execution_times, num_executions);

    printf("number of iterations = %d \n", nit);
    printf("execution time = %.10fs \n", avg_time);
    printf("\n");
    printf("our result = %.10f \n", result);
    printf("theoretical result = %.10f \n", 2.0 / N);

    free(execution_times);
    free(a_mat);
    free(b_vec);
    free(x_init);

    return 0;
}
