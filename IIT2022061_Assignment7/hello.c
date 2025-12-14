/* 
Name        : Uttkarsh Malviya 
Roll Number : IIT2022061

Experimental Setup:
- System: Running Ubuntu 22.04.3 LTS on WSL2 (Windows 10)
- CPU: 12th Gen Intel(R) Core(TM) i5-1235U, 12 logical cores (6 cores × 2 threads)
- RAM: 8 GB
- Compiler: mpicc
- Compilation Command: mpicc hello.c -o hello
- Execution Command: mpirun -np 4 ./hello

Summary:
Each process prints a “Hello World” message along with its rank and the total number of processes, verifying that multiple processes execute concurrently under MPI.

Remarks/Observations:

The output confirms parallel execution — each process prints independently.
The rank values vary from 0 to (size − 1).
Execution time is negligible (less than 1 second).
Demonstrates successful initialization and synchronization of all MPI processes.


*/


#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);                   
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    
    MPI_Comm_size(MPI_COMM_WORLD, &size);     

    printf("Hello World from process %d out of %d processes\n", rank, size);

    MPI_Finalize();                          
    return 0;
}
