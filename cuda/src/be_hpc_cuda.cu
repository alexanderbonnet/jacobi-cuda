#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 512

double *copy(double *vect, int size) {
    double *vect_new = (double *)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) {
        vect_new[i] = vect[i];
    }
    return vect_new;
}

double mean(double *x_array, int size) {
    double mean = 0;
    for (int i = 0; i < size; i++) {
        mean += x_array[i];
    }
    mean = mean / size;
    return mean;
}

double *init_x(int size) {
    double *x_vect = (double *)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) {
        x_vect[i] = 1;
    }
    return x_vect;
}

double *init_b(int size) {
    double *b_vect = (double *)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) {
        b_vect[i] = 6;
    }
    return b_vect;
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
                            double *b_vec, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size) {
        double sum = 0;
        for (int j = 0; j < size; j++) {
            if (index != j) {
                sum += a_mat[index * size + j] * x_old[j];
            }
        }
        x_new[index] = (b_vec[index] - sum) / a_mat[index * size + index];
    }
}

int solve_with_jacobi(double *x_init, double *a_mat, double *b_vec, int size,
                      double epsilon) {
    int nit = 0;
    double eps_2 = epsilon * epsilon;
    double crit = eps_2 + 1;

    double *dev_a, *dev_b, *dev_x_old, *dev_x_new, *dev_crit;

    cudaMalloc((void **)&dev_a, size * size * sizeof(double));
    cudaMalloc((void **)&dev_b, size * sizeof(double));
    cudaMalloc((void **)&dev_x_old, size * sizeof(double));
    cudaMalloc((void **)&dev_x_new, size * sizeof(double));
    cudaMalloc((void **)&dev_crit, sizeof(double));

    // copy inputs to device
    cudaMemcpy(dev_a, a_mat, size * size * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b_vec, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x_old, x_init, size * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x_new, x_init, size * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_crit, &crit, sizeof(double), cudaMemcpyHostToDevice);

    while (crit > eps_2) {
        increment_x<<<size / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            dev_x_new, dev_x_old, dev_a, dev_b, size);
        criterion<<<size / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            dev_x_new, dev_x_old, dev_crit);

        cudaMemcpy(&crit, dev_crit, sizeof(double), cudaMemcpyDeviceToHost);
        nit += 1;
        cudaMemcpy(dev_x_old, dev_x_new, size * sizeof(double),
                   cudaMemcpyDeviceToDevice);
        printf("%f\n", crit);
    }

    cudaMemcpy(x_init, dev_x_new, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_x_old);
    cudaFree(dev_x_new);
    cudaFree(dev_crit);

    return nit;
}

int main(int argc, char *argv[]) {
    int size = 1024;
    // set constants for problem size and number of executions for taking the
    // average
    int num_executions = 1;

    // number of iterations of the algorithm and result
    int nit = 0;
    double result = 0;

    double eps = 1e-6;

    double *x_init_base = init_x(size);
    double *a_mat = init_a(size);
    double *b_vec = init_b(size);

    double *execution_times = (double *)malloc(sizeof(double) * num_executions);

    struct timeval t1, t2;
    double time = 0;

    printf(
        "running sequential version of iterative Jacobi algorithm with size = "
        "%d\n",
        size);

    for (int i = 0; i < num_executions; i++) {
        double *x_init = copy(x_init_base, size);

        gettimeofday(&t1, 0);

        nit = solve_with_jacobi(x_init, a_mat, b_vec, size, eps);

        cudaDeviceSynchronize();
        gettimeofday(&t2, 0);

        time = 1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec;
        time = time / 1000.0;

        execution_times[i] = time;

        result = x_init[0];

        free(x_init);
    }

    free(execution_times);
    free(a_mat);
    free(b_vec);
    free(x_init_base);

    return 0;
}
