#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Compile Command  : gcc OMP_Block_Cyclic.c -o OMP_Block_Cyclic.out -fopenmp
// Run Command      : ./OMP_Block_Cyclic.out

#define SIZE 32

int main(int argc, char *argv[]) {
    int a[32];
    omp_set_num_threads(8);

    // == Exercise #1 ==
    // parallel block distribution to initialize to all zero
    #pragma omp parallel
    {
        int blockSize = SIZE / omp_get_num_threads();
        int tid = omp_get_thread_num();
        for (int i = tid * blockSize; i < (tid + 1) * blockSize; i++) {
            a[i] = 0;
            // Prints every 4 times a[index] is set to 0 on the 8 threads
            // printf("Value %d on thread number : %d\n", a[i], tid);
        }
    }

    // Checking output
    // printf("- ");
    // for (int i = 0; i < 32; i++) {
    //     printf("%d,", a[i]);
    // }
    // printf("\n\n");

    // == Exercise #2 ==
    // parallel block distribution to initialize to all i
    #pragma omp parallel
    {
        int blockSize = SIZE / omp_get_num_threads();
        int tid = omp_get_thread_num();
        for (int i = tid * blockSize; i < (tid + 1) * blockSize; i++) {
            a[i] = i;
            // Prints every 4 times a[index] is set to i on the 8 threads
            // printf("Value %d on thread number : %d\n", a[i], tid);
        }
    }

    // Checking output
    // printf("- ");
    // for (int i = 0; i < 32; i++) {
    //     printf("%d,", a[i]);
    // }
    // printf("\n\n");

    // == Exercise #3 ==
    // parallel cyclic distribution to count the number of even values (should be 16 including 0)
    int local_sum;
    int sum = 0;

    #pragma omp parallel shared(sum) private(local_sum)
    {
        int numThreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        local_sum = 0;

        #pragma omp parallel for
        for (int i = tid; i < SIZE; i += numThreads) {
            if (a[i] % 2 == 0) {
                local_sum++;
            }
        }

        // master thread prints total number of evens
        #pragma omp critical
        {
            sum += local_sum;
        }
    }

    // master thread prints total number of evens
    #pragma omp master
    {
        printf("\nNumber of evens (including 0) : %d\n\n", sum);
    }

    return 0;
}