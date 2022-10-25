#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// concats integer values together
int concat(int i1, int i2) {
    int temp = 10;

    while (i2 >= temp) {
        temp *= 10;
    }
    return i1 * temp + i2;
}

int createID(int rank) {
    int d = rank;
    int random = rand() % 89 + 10; // random #s between 10 and 99
    int mod = random % 2; // mod of random number and 2

    // Creates the process ID = <Random Number, Rank, Random Number % 2>
    int id;
    if (d < 10) { // if rank is less than 2 digits
        id = concat(random, 0); // append 0 to left of rank
        id = concat(id, d);
    } else { // if rank is 2 digits
        id = concat(random, d);
    }
    id = concat(id, mod);

    printf("Rank %d has ID : [%d]\n", rank, id);

    return id;
}

int main(int argc, char *argv[]) {
    int rank;
    int size;
    //arr will contain 4 elements: 
    //idx 0 will contain the smallest even number
    //idx 1 will contain the rank of the smallest even number
    //idx 2 will contain the biggest odd number 
    //idx 3 will contian the rank of the biggest odd number
    int arr[4] = {-1, -1, -1, -1}; 
    int ID; 

    // Grabs process number and passed value
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    srand(time(NULL) + rank);

    if (rank != 0) {
        ID = createID(rank);

        MPI_Recv(&arr, 4, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  
        printf("-> Rank %d has [%d].[%d] from Rank %d \n", rank, arr[0], arr[2], rank - 1);
        
        //Compare current biggest odd and smallest even with current processor
        //Check if processor is even or odd then checks greater or less
        if (ID % 2 == 0) {
            if (ID < arr[0]) {
                arr[0] = ID; 
                arr[1] = rank;
            }
        } else {
            if (ID > arr[2]) {
                arr[2] = ID;
                arr[3] = rank;
            }
        }
    } else {
        ID = createID(rank);
        if(ID % 2 == 0) { // check if even or odd and initilize the first send message
            arr[0] = ID;
            arr[1] = rank;
            arr[2] = 1;
            arr[3] = -1;
        } else {
            arr[0] = 99998;
            arr[1] = -1;
            arr[2] = ID;
            arr[3] = rank;
        }
    }

    MPI_Send(&arr, 4, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        ID = createID(rank);

        MPI_Recv(&arr, 4, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  
        printf("-> Rank %d has [%d].[%d] from Rank %d \n", rank, arr[0], arr[2], size - 1);
        
        //Compare current biggest odd and smallest even with current processor
        //Check if processor is even or odd then checks greater or less
        if (ID % 2 == 0) {
            if (ID < arr[0]) {
                arr[0] = ID; 
                arr[1] = rank;
            }
        } else {
            if (ID > arr[2]) {
                arr[2] = ID;
                arr[3] = rank;
            }
        }

        printf("\nThe President is processor %d with the ID [%d]\n", arr[1], arr[0]);
        printf("The Vice President is processor %d with the ID [%d]\n\n", arr[3], arr[2]); 
    }

    MPI_Finalize();
    return 0;
}
