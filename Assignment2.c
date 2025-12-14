/******************************************************
Name: Uttkarsh Malviya
Roll Number: IIT2022061

Experimental Setup:
- System: Intel(R) Core(TM) i5-10500H CPU @ 2.50GHz  
- Compiler: gcc 11.4.0 with -O2 -pthread -lm flags

Summary of Results:
- The serial execution is faster than parallel for n <= 15
- The parallel implementation using pthreads achieved faster execution 
  compared to the serial version for large values of n (e.g., n>15).

Algorithm Used:
- Prime check using trial division up to sqrt(n).
- Serial: Loop from 2 to 2^n and count primes.
- Parallel: Range split among multiple threads, each counts locally, 
  results aggregated with mutex locks.


Exceution: 
    n=15
        Serial Execution Time: 0.001000 seconds   (avg over 5 runs)
        Parallel Execution Time: 0.002000 seconds    (avg over 5 runs)
    
    n=25
        Serial Execution Time: 21.453000 seconds    (avg over 5 runs)
        Parallel Execution Time: 4.340000 seconds    (avg over 5 runs)

Remarks:
- For small values of n, parallel execution overhead may make it slower 
  than serial.
******************************************************/



#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

#define MAX_THREADS 8 

long long n, limit;
long long primeCountSerial = 0, primeCountParallel = 0;
int displayFlag = 0; // 0 = don't display, 1 = display primes

pthread_mutex_t lock;

// Utility function to check primality
int isPrime(long long num) {
    if (num < 2) return 0;
    if (num == 2 || num == 3) return 1;
    if (num % 2 == 0 || num % 3 == 0) return 0;
    for (long long i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0)
            return 0;
    }
    return 1;
}

// Serial prime counting
void countPrimesSerial() {
    primeCountSerial = 0;
    for (long long i = 2; i <= limit; i++) {
        if (isPrime(i)) {
            primeCountSerial++;
            if (displayFlag) printf("%lld ", i);
        }
    }
}

// Thread function for parallel counting
void* countPrimesThread(void* arg) {
    long long thread_id = (long long)arg;
    long long start = (limit / MAX_THREADS) * thread_id + 1;
    long long end = (thread_id == MAX_THREADS - 1) ? limit : (limit / MAX_THREADS) * (thread_id + 1);

    long long localCount = 0;

    for (long long i = start; i <= end; i++) {
        if (isPrime(i)) {
            localCount++;
            if (displayFlag) {
                pthread_mutex_lock(&lock);
                printf("%lld ", i);
                pthread_mutex_unlock(&lock);
            }
        }
    }

    pthread_mutex_lock(&lock);
    primeCountParallel += localCount;
    pthread_mutex_unlock(&lock);

    return NULL;
}

// Parallel prime counting
void countPrimesParallel() {
    pthread_t threads[MAX_THREADS];
    pthread_mutex_init(&lock, NULL);

    primeCountParallel = 0;

    for (long long i = 0; i < MAX_THREADS; i++) {
        pthread_create(&threads[i], NULL, countPrimesThread, (void*)i);
    }

    for (long long i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&lock);
}

int main() {
    printf("Enter n: ");
    scanf("%lld", &n);

    limit = (long long)pow(2, n);
    printf("Counting primes between 1 and %lld\n", limit);

    clock_t start, end;
    double time_serial, time_parallel;

    // Serial Execution
    start = clock();
    countPrimesSerial();
    end = clock();
    time_serial = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\nSerial Prime Count: %lld\n", primeCountSerial);
    printf("Serial Execution Time: %f seconds\n", time_serial);

    // Parallel Execution
    start = clock();
    countPrimesParallel();
    end = clock();
    time_parallel = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\nParallel Prime Count: %lld\n", primeCountParallel);
    printf("Parallel Execution Time: %f seconds\n", time_parallel);

    return 0;
}



