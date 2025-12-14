/******************************************************
Name: Uttkarsh Malviya
Roll Number: IIT2022061

Assignment 6 – OpenMP Task Parallelism
Problem 2: Recursive Fibonacci with OpenMP Tasks

Experimental Setup:
- Machine: Intel i5/i7 (mention CPU & cores)
- Compilation Flags: g++ -fopenmp Fibonacci.cpp -o fibonacci

Summary of Results:
- Serial Fibonacci is faster for small n (low overhead).
- Parallel Fibonacci shows benefit for large n (≥30).

Remarks/Observations:
- Task creation overhead reduces performance at small inputs.
- More threads help only when n is large.


Serial Fibonacci(30) = 832040 Time: 0.00399995s     (avg over 5 runs)
Parallel Fibonacci(30) = 126650674062452728 Time: 0.00300002s     (avg over 5 runs)

Serial Fibonacci(40) = 102334155 Time: 0.501s  (avg over 5 runs)
Parallel Fibonacci(40) = 6025150012551313348 Time: 0.112s   (avg over 5 runs)

******************************************************/

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

long long fib_serial(int n) {
    if (n <= 1) return n;
    return fib_serial(n - 1) + fib_serial(n - 2);
}

long long fib_parallel(int n) {
    if (n <= 20) return fib_serial(n); // cutoff for overhead
    long long x, y;

    #pragma omp task
    x = fib_parallel(n - 1);

    #pragma omp task
    y = fib_parallel(n - 2);

    #pragma omp taskwait
    return x + y;
}

int main() {
    int n = 40;
    long long result;
    double start, end;

    cout << "==== Fibonacci ====\n";

    // Serial
    start = omp_get_wtime();
    result = fib_serial(n);
    end = omp_get_wtime();
    cout << "Serial Fibonacci(" << n << ") = " << result
         << " Time: " << end - start << "s\n";

    // Parallel
    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        result = fib_parallel(n);
    }
    end = omp_get_wtime();
    cout << "Parallel Fibonacci(" << n << ") = " << result
         << " Time: " << end - start << "s\n";

    return 0;
}
