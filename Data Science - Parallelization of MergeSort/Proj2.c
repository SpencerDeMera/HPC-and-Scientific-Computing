#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Compile Command  : gcc Proj2.c -o exec -fopenmp
// Run Command      : ./exec

#define X 2048

// Serial Merge Function
void merge(int* arr, int l, int mid, int r) {
    int i, j, k;
    int n1 = mid - l + 1;
    int n2 = r - mid;

    // create temp arrays
    int *L = malloc(n1 * sizeof(int));
    int *R = malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++) {
        L[i] = arr[l + i];
    }
    for (j = 0; j < n2; j++) {
        R[j] = arr[mid + 1 + j];
    }

    // Merge the temp arrays back into arr[l..r]
    i = 0;          // Initial index of first subarray
    j = 0;          // Initial index of second subarray
    k = l;          // Initial index of merged subarray

    // Compare and reorganize Left and Right temp arrays
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Parallel Merge Function
// l is for left index and r is right index of the sub-array of arr to be sorted
void mergeSort(int* arr, int l, int r) {
    if (l < r) {
        // Same as (l+r)/2, but avoids overflow for large l and h
        int mid = l + (r - l) / 2;

        #pragma omp task shared(arr)
        {
            mergeSort(arr, l, mid);

            // FOR DEBUGGING:
            // printf("%d", omp_get_thread_num());
        }

        #pragma omp task shared(arr)
        {
            mergeSort(arr, mid + 1, r);

            // FOR DEBUGGING:
            // printf("%d", omp_get_thread_num());
        }

        // wait for all threads to finished
            // Acts like a lock
        #pragma omp taskwait
        merge(arr, l, mid, r);
    }
}

// Initalizes array with random values
void initArr(int* mainArr, int SIZE) {
    srand(time(NULL));

    for (int i = 0; i < SIZE; i++) {
        mainArr[i] = rand() % X;
    }
}

void printArr(int* mainArr, int SIZE) {
    for (int i = 0; i < SIZE; i++) {
        printf("%d ", mainArr[i]);
    }
    printf("\n\n");
}

void copyArr(int* mainArr, int* multiThreadArr, int SIZE) {
    for (int i = 0; i < SIZE; i++) {
        multiThreadArr[i] = mainArr[i];
    }
}

int main() {
    int SIZE = 0;
    double start1, start2;
    double end1, end2;
    double time1, time2;

    printf("\nDesired Array Length: ");
    scanf("%d", &SIZE);
    printf("\n");

    int *mainArr = malloc(SIZE * sizeof(int));
    int *_2ThreadArr = malloc(SIZE * sizeof(int));
    int *_4ThreadArr = malloc(SIZE * sizeof(int));

    // Initialize array of random values
    initArr(mainArr, SIZE);

    // Copy to do additional temporary arrays
    copyArr(mainArr, _2ThreadArr, SIZE);
    copyArr(mainArr, _4ThreadArr, SIZE);

    // Sorting on N number of threads (N = 2)
    omp_set_num_threads(2);

    start1 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(_2ThreadArr, 0, SIZE - 1);
    }
    end1 = omp_get_wtime();

    // Sorting on N number of threads (N = 4)
    omp_set_num_threads(4);

    start2 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(_4ThreadArr, 0, SIZE - 1);
    }
    end2 = omp_get_wtime();

    // Print statements for all three arrays
    printf("\nInitial Array:\n");
    printArr(mainArr, SIZE);

    printf("\nSorted Array:\n");
    printArr(_2ThreadArr, SIZE);

    printf("\nSorted Array:\n");
    printArr(_4ThreadArr, SIZE);

    // Output elapsed time in seconds for both runs
    time1 = end1 - start1;
    printf("Sorted Array on 2 threads in %f seconds\n", time1);
    time2 = end2 - start2;
    printf("Sorted Array on 4 threads in %f seconds\n", time2);

    printf("\nProgram Finished...\n\n");
    return 0;
}
