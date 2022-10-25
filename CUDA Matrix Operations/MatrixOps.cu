#include <stdio.h>
#include <cuda.h>
#include <time.h>

// Compile  : nvcc MatrixOps.cu -o MatrixOps.out
// Run      : ./MatrixOps.out

#define N 8
#define SIZE 16
#define BLOCK 4

// Exercise 1
__global__ void evensCheck(unsigned *matrix, int *ctrArr) {
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < N * N) { // bound checking for within 64 elements
        if (matrix[id] % 2 == 0) {
            ctrArr[id] += 1;
        } else {
            ctrArr[id] += 0;
        }
    }
}

// Exercise 1
__host__ int findEvens(unsigned *hmatrix) {
    unsigned *dmatrix;
    int *hctrArr, *dctrArr;
    int sumEvens = 0;
    
    cudaMalloc(&dmatrix, N * N * sizeof(unsigned)); // alloc device memory for device matrix
    cudaMalloc(&dctrArr, N * N * sizeof(unsigned)); // alloc device memory for device even counter array
    hctrArr = (int *)malloc(N * N * sizeof(int)); // alloc host memory for device even counter array

    // Copies host matrix to device matrix
    cudaMemcpy(dmatrix, hmatrix, N * N * sizeof(unsigned), cudaMemcpyHostToDevice);
    // Calls CUDA __global__ evensCheck function to find number of even values
    evensCheck<<<BLOCK, SIZE>>>(dmatrix, dctrArr);
    // Copies device matrix to host matrix
    cudaMemcpy(hctrArr, dctrArr, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < (N * N); i++) {
        sumEvens += hctrArr[i]; // sums host ctrArr (1 if even && 0 if odd)
    }
    return sumEvens;
}

// Exercise 2
__global__ void squareMatrix(unsigned *matrix) {
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < N * N) { // bound checking for within 64 elements
        matrix[id] *= matrix[id];
    }
}

// Exercise 2
__host__ void findSquare(unsigned *hmatrix) {
    unsigned *dmatrix;
    
    cudaMalloc(&dmatrix, N * N * sizeof(unsigned)); // alloc device memory for device matrix

    // Copies host matrix to device matrix
    cudaMemcpy(dmatrix, hmatrix, N * N * sizeof(unsigned), cudaMemcpyHostToDevice);
    // Calls CUDA __global__ squareMatrix function to square matrix values
    squareMatrix<<<BLOCK, SIZE>>>(dmatrix);
    // Copies device matrix to host matrix
    cudaMemcpy(hmatrix, dmatrix, N * N * sizeof(int), cudaMemcpyDeviceToHost);
}

__global__ void init(unsigned *matrix) {
    unsigned id = threadIdx.x * blockDim.y + threadIdx.y;
    if (id < N * N) {
        matrix[id] = id;
    }
}

int main() {
    dim3 block(N, N, 1);
    unsigned *dmatrix, *hmatrix;
    
    cudaMalloc(&dmatrix, N * N * sizeof(unsigned));
    hmatrix = (unsigned *)malloc(N * N * sizeof(unsigned));

    // Calls CUDA kernel init function to init matrix with values
        // Values are just tid of each grid spot
    init<<<1, block>>>(dmatrix);

    // Copy `matrix` from GPU memory to `hmatrix` on Host CPU memory
    cudaMemcpy(hmatrix, dmatrix, N * N * sizeof(unsigned), cudaMemcpyDeviceToHost);

    // Print out the N * N matrix `hmatrix`
    printf("\nThe Matrix:\n");
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            printf("%2d ", hmatrix[i * N + j]);
        }
        printf("\n");
    }

    // Exercise # 1
    // Calls findEvens host function
    int numEvens = findEvens(hmatrix);
    printf("\nNumber of Evens (including 0) : %d\n", numEvens);

    // Exercise # 2
    // Calls findSquare host function
    findSquare(hmatrix);

    // print out the square matrix
    printf("\nThe Squared Matrix:\n");
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            printf("%2d ", hmatrix[i * N + j]);
        }
        printf("\n");
    }

    // Syncs CPU and GPU
    cudaDeviceSynchronize();

    printf("\nProgram Ending...\n");
    return 0;
}