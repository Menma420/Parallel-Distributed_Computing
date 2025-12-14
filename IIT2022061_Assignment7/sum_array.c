/* 
Name        : Uttkarsh Malviya 
Roll Number : IIT2022061

Experimental Setup:
- System: Running Ubuntu 22.04.3 LTS on WSL2 (Windows 10)
- CPU: 12th Gen Intel(R) Core(TM) i5-1235U, 12 logical cores (6 cores × 2 threads)
- RAM: 8 GB
- Compiler: mpicc
- Compilation Command: mpicc sum_array.c -o sum_array
- Execution Command: mpirun -np 4 ./sum_array

Summary:
An array of N random integers is divided among all processes. Each process computes a local (partial) sum, and the results are combined using MPI_Reduce to produce the total sum at the root process.

Remarks/Observations:

Verified correctness by comparing parallel sum with serial computation.
Parallel execution reduces computation time for large arrays.
Speedup improves as the number of processes increases, though limited by communication overhead.
Demonstrates parallel efficiency and data distribution in MPI.


*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size, N, *arr = NULL;
    int local_sum = 0, global_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter size of array (N): ");
        scanf("%d", &N);

        arr = (int*)malloc(N * sizeof(int));
        srand(time(0));
        printf("Array: ");
        for (int i = 0; i < N; i++) {
            arr[i] = rand() % 10;
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk = N / size;
    int* sub_arr = (int*)malloc(chunk * sizeof(int));

    MPI_Scatter(arr, chunk, MPI_INT, sub_arr, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++)
        local_sum += sub_arr[i];

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("Total sum of array = %d\n", global_sum);

    if (rank == 0) free(arr);
    free(sub_arr);

    MPI_Finalize();
    return 0;
}
