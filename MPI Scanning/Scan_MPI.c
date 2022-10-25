#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

// FIXED
// Compile Command  : mpicc Scan_MPI.c -o Scan_MPI.out
// Run Command      : mpirun --oversubscribe -np 8 ./Scan_MPI.out

int main(int argc, char *argv[]) {
    int rank;
    int size;
    int d;
    int total;
    // int a[8] = {3, 1, 7, 1, 4, 1, 6, 3}; // if you use hardcoded values it works great

    srand(time(NULL));

    // Grabs process number and passed value
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    d = rank;
    int random = (rand() % 25) + 1; // Random number between 1 and 25 inclusive
    // printf("value at process %d is : %d\n", d, random);

    // Result = Result + Process #
    // Total of process 3 : (total of process 2) + 3 = 5
    MPI_Scan(&random, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // Removed unecessary Send and Recv operators and/in if-checks
        // Originally did this but incorrectly printed so I turned in previous file
    // MPI_Scan(&a, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); // version for hardcoded array values

    printf("- Process %d will recieve the array portion between index %d-%d\n", d, ((total - random) + 1), total);

    MPI_Finalize();
    return 0;
}