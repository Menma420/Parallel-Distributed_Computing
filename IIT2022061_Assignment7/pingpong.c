/* 
Name        : Uttkarsh Malviya 
Roll Number : IIT2022061

Experimental Setup:
- System: Running Ubuntu 22.04.3 LTS on WSL2 (Windows 10)
- CPU: 12th Gen Intel(R) Core(TM) i5-1235U, 12 logical cores (6 cores × 2 threads)
- RAM: 8 GB
- Compiler: mpicc
- Compilation Command: mpicc pingpong.c -o pingpong
- Execution Command: mpirun -np 4 ./pingpong

Summary:
Two MPI processes (Process 0 and Process 1) exchange an integer message — Process 0 sends (“Ping”) and Process 1 replies (“Pong”) — to demonstrate inter-process communication using MPI_Send and MPI_Recv.

Remarks/Observations:

Message transmission and reception work correctly in both directions.
Verified that the received integer remains consistent after the round trip.
Communication latency is minimal; suitable for small message exchanges.
Confirms reliable point-to-point communication in MPI.


*/

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, msg;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        msg = 64; 
        printf("Process 0 sending message %d to Process 1\n", msg);
        MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);  

        MPI_Recv(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
        printf("Process 0 received message back: %d\n", msg);
    }

    else if (rank == 1) {
        MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status); 
        printf("Process 1 received message: %d\n", msg);

        MPI_Send(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); 
    }

    MPI_Finalize();
    return 0;
}
