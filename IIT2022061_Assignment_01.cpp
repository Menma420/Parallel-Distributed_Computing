/*

Name: Uttkarsh Malviya
Roll No: IIT2022061

Experimental Setup: Intel(R) Core(TM) i5-10500H CPU @ 2.50GHz, 6 cores (12 threads), 8GB RAM, Windows 11

Execution Times (in ms, averaged over 5 runs):
1 Thread:   0.89
2 Threads:  0.81
4 Threads:  0.77
8 Threads:  0.72
16 Threads: 1.05

Summary: The execution time generally decreases as we increase the number of threads,
showing improved performance due to parallel computation up to an optimal level. Beyond
8 threads, speedup saturates due to overheads and small problem size.
Since the maximum number of threads is 12 in my setup, using more than 12 threads decreases 
performance due to context switching and overhead instead of improving it.

*/

#include <iostream>
#include <pthread.h>
#include <chrono>
#include <vector>

const int N = 1024;
double array[N];
int M;  // Number of threads
double* partial_sums;

// Structure to pass to threads
struct ThreadData {
    int thread_id;
    int start_idx;
    int end_idx;
};

// Thread function
void* thread_sum(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    double sum = 0.0;
    for (int i = data->start_idx; i < data->end_idx; ++i)
        sum += array[i];
    partial_sums[data->thread_id] = sum;
    pthread_exit(nullptr);

    return nullptr;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./sum <num_threads>\n";
        return 1;
    }

    M = std::stoi(argv[1]);
    if (M <= 0 || M > N) {
        std::cerr << "Error: Invalid number of threads.\n";
        return 1;
    }

    // Initialize the array
    for (int i = 0; i < N; ++i)
        array[i] = i;

    partial_sums = new double[M];
    pthread_t* threads = new pthread_t[M];
    ThreadData* thread_data = new ThreadData[M];

    int chunk_size = N / M;
    int remainder = N % M;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create threads
    for (int i = 0; i < M; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].start_idx = i * chunk_size;
        thread_data[i].end_idx = (i == M - 1) ? (i + 1) * chunk_size + remainder : (i + 1) * chunk_size;
        pthread_create(&threads[i], nullptr, thread_sum, (void*)&thread_data[i]);
    }

    // Wait for threads
    for (int i = 0; i < M; ++i)
        pthread_join(threads[i], nullptr);

    // Combine partial sums
    double total_sum = 0.0;
    for (int i = 0; i < M; ++i)
        total_sum += partial_sums[i];

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    double formula_sum = (double)N * (N - 1) / 2;

    std::cout << "N: " << N << "\n";
    std::cout << "Threads: " << M << "\n";
    std::cout << "Loop Sum: " << total_sum << "\n";
    std::cout << "Formula Sum: " << formula_sum << "\n";
    std::cout << "Execution Time (ms): " << elapsed.count() << "\n";

    delete[] threads;
    delete[] thread_data;
    delete[] partial_sums;

    return 0;
}


