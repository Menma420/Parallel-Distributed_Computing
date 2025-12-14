# Parallel & Distributed Computing Labs

## Overview
This repository contains a collection of high-performance computing (HPC) implementations exploring concurrency, memory optimization, and hardware acceleration. The projects focus on minimizing latency and maximizing throughput using **C++, CUDA, OpenMP, MPI, and Pthreads**.

Key areas of focus include **GPU shared memory optimization**, **sparse matrix storage formats (CSR)**, and **distributed message passing**.

## 🛠️ Technologies & Tools
* **Languages:** C++, C
* **Parallel Frameworks:** CUDA, OpenMP, MPI, Pthreads
* **Concepts:** SIMD, Shared Memory Tiling, Task Parallelism, Race Condition Handling, Load Balancing

## 🚀 Key Implementations

### 1. CUDA & GPU Acceleration
* **Sobel Edge Detection (Shared Memory):**
    * Implemented a convolution kernel for edge detection on 64x64 grayscale images.
    * **Optimization:** Leveraged **Shared Memory tiling** to minimize global memory access latency and improve bandwidth usage.
    * Benchmarked performance across varying block sizes (8x8, 16x16, 32x32).
* **Vector Operations:** Parallelized vector addition and scalar multiplication using massive thread dispatching.

### 2. Multi-Threading with Pthreads
* **Sparse Matrix-Vector Multiplication (SpMV):**
    * Implemented the **Compressed Sparse Row (CSR)** format to optimize memory footprint and cache locality for sparse data.
    * Engineered a multi-threaded multiplication engine to handle large-scale linear algebra operations efficiently.
* **Parallel Prime Sieve:** Counted primes in large ranges using domain decomposition and dynamic load balancing.

### 3. OpenMP Tasking & Data Parallelism
* **Weather Data Analysis:**
    * Processed large-scale meteorological datasets to compute global min/max and averages.
    * Utilized **Reduction clauses** and **Critical sections** to ensure thread safety without compromising speedup.
* **Task-Based Merge Sort:**
    * Implemented dynamic task parallelism using `#pragma omp task` to parallelize recursive sorting algorithms.

### 4. Distributed Computing (MPI)
* **Distributed Array Sum:**
    * Utilized Message Passing Interface (MPI) to distribute array processing across multiple nodes/processes.
* **Ping-Pong Communication:**
    * modeled blocking point-to-point communication latency between distributed processes.

## 📊 Benchmarks
* **Speedup Analysis:** Achieved **~3x speedup** on quad-core CPUs for Merge Sort implementations.
* **Amdahl's Law:** Analyzed theoretical vs. actual speedup limits across 1-32 thread counts to identify overhead bottlenecks.

## 🔧 Build & Run
**CUDA Programs:**
```bash
nvcc -o edge_detect edge_detection.cu
./edge_detect
````

**OpenMP Programs:**

```bash
g++ -fopenmp -o mergesort merge_sort.cpp
./mergesort
```

**MPI Programs:**

```bash
mpicc -o mpi_sum distributed_sum.c
mpirun -np 4 ./mpi_sum
```

-----

*Author: Uttkarsh Malviya*
