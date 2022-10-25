#include <stdio.h>
#include <cuda.h>

// Compile  : nvcc GPU_Kernel_Ops.cu -o GPU_Kernel_Ops.out
// Run      : ./GPU_Kernel_Ops.out

#define BLOCKSIZE 8

// Exercise(s) 1 & 2
__global__ void kernelInit(int *dArray, int SIZE) {
    int id =  blockIdx.x * blockDim.x + threadIdx.x;
    if (id < SIZE) { // if tid < SIZE
        for (int i = 0; i < SIZE; i++) {
            dArray[id] = 0; // inits all on each block to 0
        }
    }
}

// Exercise 3
__global__ void kernelAdd(int *dArray, int SIZE) {
    int id =  blockIdx.x * blockDim.x + threadIdx.x;
    if (id < SIZE) { // if tid < SIZE
        for (int i = 0; i < SIZE; i++) {
            dArray[id] = id; // inits all on each block to tid
        }
    }
}

int main() {
    int *hostArray1;
    int *deviceArray1;
    int *hostArray2;
    int *deviceArray2;
    // int SIZE = 32; // Exercise 1
    // int SIZE = 1024; // Exercise 2
    int SIZE = 8192; // Exercise 4

    // Exercise(s) 1 & 2
    cudaMalloc(&deviceArray1, SIZE * sizeof(int)); // allocated GPU space
    hostArray1 = (int *)malloc(SIZE * sizeof(int));

    int numBlocks = ceil((int) SIZE / BLOCKSIZE);
    printf("Num Blocks = %d\n", numBlocks); // gets number of blocks in total

    // numBlocks number of blocks across a BLOCKSIZE of 8
    kernelInit<<<numBlocks, BLOCKSIZE>>>(deviceArray1, SIZE); // Exercise 1 & 2
    cudaMemcpy(hostArray1, deviceArray1, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; ++i) {
        printf("%d, ", hostArray1[i]);
    }
    printf("\n\n");

    // Exercise 3
    cudaMalloc(&deviceArray2, SIZE * sizeof(int)); // allocated GPU space
    hostArray2 = (int *)malloc(SIZE * sizeof(int));

    numBlocks = ceil((int) SIZE / BLOCKSIZE);
    printf("Num Blocks = %d\n", numBlocks); // gets number of blocks in total

    // numBlocks number of blocks across a BLOCKSIZE of 8
    kernelAdd<<<numBlocks, BLOCKSIZE>>>(deviceArray2, SIZE);
    cudaMemcpy(hostArray2, deviceArray2, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; ++i) {
        printf("%d, ", hostArray2[i]);
    }
    printf("\n");

    return 0;
}