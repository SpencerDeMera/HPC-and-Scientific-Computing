#include <iostream>
#include <stdio.h>
#include <mpi.h>
using namespace std;

// MPI Blocking
void blocking(int rank, int size) {
    float msg;

    if (rank == 0) {
        msg = -1.f;

        MPI_Send(&msg, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        printf("Blocking Process 0 sent msg %f to process 1\n", msg);
    } else if (rank == 1) {
        MPI_Recv(&msg, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Blocking Process 1 received msg %f from process 0\n", msg);
    }
}

// MPI Non-Blocking
void nonBlocking(int rank, int size) {
    float msg;

    MPI_Request request;
    MPI_Status status;

    if (rank == 0) {
        msg = -1.f;

        MPI_Isend(&msg, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &request);
        printf("Non-Blocking Process 0 sent msg %f to process 1\n", msg);
        MPI_Wait(&request, &status);
    } else if (rank == 1) {
        MPI_Irecv(&msg, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);
        printf("Non-Blocking Process 1 received msg %f from process 0\n", msg);
    }
}

// MPI Two Blocking
void twoBlocking(int rank, int size) {
    float msg1;

    if (rank == 0) {
        msg1 = -1.f;

        MPI_Send(&msg1, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        printf("Two Blocking Process 0 sent msg %f to process 1\n", msg1);

        MPI_Recv(&msg1, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Two Blocking Process 0 received msg %f from process 1\n", msg1);
    } else if (rank == 1) {
        MPI_Recv(&msg1, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Two Blocking Process 1 received msg %f from process 0\n", msg1);

        MPI_Send(&msg1, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        printf("Two Blocking Process 1 sent msg %f to process 0\n", msg1);
    }
}

// MPI Two Non-Blocking
void twoNonBlocking(int rank, int size) {
    float msg1;

    MPI_Request request;
    MPI_Status status;

    if (rank == 0) {
        msg1 = -1.f;

        MPI_Isend(&msg1, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &request);
        printf("Two Non-Blocking Process 0 sent msg %f to process 1\n", msg1);
        MPI_Wait(&request, &status);

        MPI_Irecv(&msg1, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);
        printf("Two Non-Blocking Process 0 received msg %f from process 1\n", msg1);
    } else { 
        MPI_Irecv(&msg1, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);
        printf("Two Non-Blocking Process 1 received msg %f from process 0\n", msg1);

        MPI_Isend(&msg1, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request);
        printf("Two Non-Blocking Process 1 sent msg %f to process 0\n", msg1);
        MPI_Wait(&request, &status);
    }
}

int main() {
    int rank;
    int size;
    double bt1, bt2;
    double nbt1, nbt2;
    double tbt1, tbt2;
    double tnbt1, tnbt2;

    // init MPI commands
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Single Transmission
    bt1 = MPI_Wtime();
    blocking(rank, size);
    bt2 = MPI_Wtime();

    nbt1 = MPI_Wtime();
    nonBlocking(rank, size);
    nbt2 = MPI_Wtime();

    // Two Transmission
    tbt1 = MPI_Wtime();
    twoBlocking(rank, size);
    tbt2 = MPI_Wtime();

    tnbt1 = MPI_Wtime();
    twoNonBlocking(rank, size);
    tnbt2 = MPI_Wtime();

    MPI_Finalize();

    if (rank == 0) {
        cout << endl;
        cout << "- One Trip Blocking Time Elapsed: " << bt2 - bt1 << endl;
        cout << "- One Trip Non-Blocking Time Elapsed: " << nbt2 - nbt1 << endl;
        cout << "- Two Trip Blocking Time Elapsed: " << tbt2 - tbt1 << endl;
        cout << "- Two Trip Non-Blocking Time Elapsed: " << tnbt2 - tnbt1 << endl;
    }
    return 0;
}