#include <iostream>
#include <stdio.h>
#include <mpi.h>
using namespace std;

int main() {
    int rank;
    int size;
    int token;
    double t1, t2;

    // init MPI commands
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    t1 = MPI_Wtime();

    if (rank != 0) {
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received token %d from process %d\n", rank, token, rank - 1);
    } else {
        token = 7;
    }

    MPI_Send(&token, 1, MPI_INT, (rank + 1)  % size, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received token %d from process %d\n", rank, token, size - 1);
    }
    t2 = MPI_Wtime();

    MPI_Finalize();

    if (rank == 0) {
        cout << endl;
        cout << "- Time Elapsed: " << t2 - t1 << "s\n";
    }
    return 0;
}