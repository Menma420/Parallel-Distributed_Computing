/*
Assignment 5 — OpenMP Weather Data Analysis

Name: Uttkarsh Malviya
Roll No.: IIT2022061


Experimental Setup:
    Machine: 12th Gen Intel(R) Core(TM) i5-1235U @ 1.30GHz, 6 cores / 12 threads, 3.7 GB RAM (available to WSL2)
    OS & Compiler: Ubuntu 22.04.3 LTS running on WSL2 (Windows host), GCC 11.4.0 with OpenMP support

Summary of Results (fill after running):
    Sequential time: ~0.0023 s
    Parallel time (4 threads): ~0.00198 s
    Speedup: ~1.16× (0.0023 / 0.00198)
    Days > 40 °C: 362,496
    Global Maximum Temperature: 50.00 °C
    Global Minimum Temperature: 10.00 °C

Observations (fill after running):
    Parallel execution gave only modest speedup for 1M records due to thread overhead.=
    At 2 threads, performance was slower than sequential.
    Best speedup was ~1.6× at 8 threads, but efficiency dropped with more threads.
    File I/O is sequential and unaffected by OpenMP.
    Larger datasets would show better parallel efficiency.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <omp.h>

/*********************** CONFIGURABLE MACROS **********************************/
#ifndef NUM_CITIES
#define NUM_CITIES   10          // number of cities
#endif
#ifndef NUM_RECORDS
#define NUM_RECORDS  1000000     // total lines in input.txt
#endif

// temperature range for generation
#ifndef TEMP_MIN
#define TEMP_MIN     10.0
#endif
#ifndef TEMP_MAX
#define TEMP_MAX     50.0
#endif

#define INPUT_FILE   "input.txt"

/*********************** UTILITY **********************************************/
static inline double drand01(unsigned int *seed) {
    // Thread-safe-ish: use local seed passed by value (OpenMP private copy).
    // xorshift32
    unsigned int x = *seed;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; *seed = x;
    // Scale to [0,1)
    return (x & 0xFFFFFF) / (double)0x1000000;
}

static void die(const char *msg) {
    fprintf(stderr, "Error: %s\n", msg);
    exit(EXIT_FAILURE);
}

/*********************** TASK 1: DATASET GENERATION ***************************/
void generate_input_file(void) {
    FILE *fp = fopen(INPUT_FILE, "w");
    if (!fp) die("cannot open input.txt for writing");

    // Use a base seed from time; per-record, we can keep one RNG (fast enough)
    unsigned int seed = (unsigned int)time(NULL) ^ 0xA5A5A5A5U;

    for (long long i = 0; i < (long long)NUM_RECORDS; ++i) {
        // Random CityID in [1, NUM_CITIES]
        int city = (int)(drand01(&seed) * NUM_CITIES) + 1;
        if (city > NUM_CITIES) city = NUM_CITIES; // guard edge

        // Biased temperature: mix a mild base + occasional heat spikes
        double u = drand01(&seed);
        double temp = TEMP_MIN + (TEMP_MAX - TEMP_MIN) * u;
        // add localized heatwave pulses to create >40°C days
        if (drand01(&seed) < 0.15) {
            double bump = 5.0 * drand01(&seed); // small surge
            temp = (temp + 38.0 + bump);
            if (temp > TEMP_MAX) temp = TEMP_MAX - drand01(&seed) * 0.1;
        }
        fprintf(fp, "%d\t%.2f\n", city, temp);
    }

    fclose(fp);
    printf("Generated %d records for %d cities into %s\n", NUM_RECORDS, NUM_CITIES, INPUT_FILE);
}

/*********************** COMMON: DATA LOADING *********************************/
typedef struct { int city; float temp; } Record;

// Load file fully into memory to allow parallel per-record processing
// Returns number of records loaded (should be NUM_RECORDS), and allocates arrays
long long load_dataset(const char *path, int **city_ids_out, float **temps_out) {
    FILE *fp = fopen(path, "r");
    if (!fp) die("cannot open input.txt for reading — run with --generate first");

    // Pre-allocate expected size; will grow if needed (rare if macros match)
    long long cap = (long long)NUM_RECORDS;
    if (cap < 1024) cap = 1024;
    int *city_ids = (int*) malloc(cap * sizeof(int));
    float *temps   = (float*) malloc(cap * sizeof(float));
    if (!city_ids || !temps) die("malloc failed");

    long long n = 0;
    int city; double t;
    while (fscanf(fp, "%d%lf", &city, &t) == 2) {
        if (n >= cap) {
            cap = (long long)(cap * 1.5) + 1024;
            city_ids = (int*)   realloc(city_ids, cap * sizeof(int));
            temps    = (float*) realloc(temps,    cap * sizeof(float));
            if (!city_ids || !temps) die("realloc failed");
        }
        city_ids[n] = city;
        temps[n]    = (float)t;
        ++n;
    }
    fclose(fp);

    *city_ids_out = city_ids;
    *temps_out    = temps;
    return n;
}

/*********************** TASK 2: SEQUENTIAL IMPLEMENTATION ********************/
typedef struct {
    double global_max;
    double global_min;
    long long hot_days; // count temp > 40
    double city_sums[NUM_CITIES];
    long long city_counts[NUM_CITIES];
} Result;

void sequential_analysis(const int *city_ids, const float *temps, long long N, Result *out) {
    Result R;
    R.global_max = -DBL_MAX;
    R.global_min = DBL_MAX;
    R.hot_days   = 0;
    for (int c = 0; c < NUM_CITIES; ++c) { R.city_sums[c] = 0.0; R.city_counts[c] = 0; }

    double t0 = omp_get_wtime();
    for (long long i = 0; i < N; ++i) {
        int c = city_ids[i] - 1; // city IDs are 1..NUM_CITIES
        if (c < 0 || c >= NUM_CITIES) continue; // skip malformed rows
        double temp = temps[i];

        if (temp > R.global_max) R.global_max = temp;
        if (temp < R.global_min) R.global_min = temp;
        if (temp > 40.0) R.hot_days++;
        R.city_sums[c]   += temp;
        R.city_counts[c] += 1;
    }
    double t1 = omp_get_wtime();

    *out = R;
    printf("Sequential Execution Time: %.6f seconds\n", t1 - t0);
}

/*********************** TASK 3: PARALLEL IMPLEMENTATION (OpenMP) *************/
#ifndef USE_ARRAY_REDUCTION
// Auto-enable array-section reduction if compiler supports OpenMP 4.5 (201511)
#  if defined(_OPENMP) && (_OPENMP >= 201511)
#    define USE_ARRAY_REDUCTION 1
#  else
#    define USE_ARRAY_REDUCTION 0
#  endif
#endif

void parallel_analysis(const int *city_ids, const float *temps, long long N, int num_threads, Result *out) {
    Result R;
    R.global_max = -DBL_MAX;
    R.global_min = DBL_MAX;
    R.hot_days   = 0;
    for (int c = 0; c < NUM_CITIES; ++c) { 
        R.city_sums[c] = 0.0; 
        R.city_counts[c] = 0; 
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(num_threads)
    {
        double local_max = -DBL_MAX;
        double local_min = DBL_MAX;
        long long local_hot = 0;
        double local_sums[NUM_CITIES];
        long long local_counts[NUM_CITIES];
        for (int c = 0; c < NUM_CITIES; ++c) { 
            local_sums[c] = 0.0; 
            local_counts[c] = 0; 
        }

        #pragma omp for nowait
        for (long long i = 0; i < N; ++i) {
            int c = city_ids[i] - 1;
            if ((unsigned)c >= NUM_CITIES) continue;
            double temp = temps[i];
            if (temp > 40.0) local_hot++;
            if (temp > local_max) local_max = temp;
            if (temp < local_min) local_min = temp;
            local_sums[c]   += temp;
            local_counts[c] += 1;
        }

        #pragma omp critical
        {
            if (local_max > R.global_max) R.global_max = local_max;
            if (local_min < R.global_min) R.global_min = local_min;
            R.hot_days += local_hot;
            for (int c = 0; c < NUM_CITIES; ++c) {
                R.city_sums[c]   += local_sums[c];
                R.city_counts[c] += local_counts[c];
            }
        }
    }

    double t1 = omp_get_wtime();

    *out = R;
    printf("Parallel Execution Time (%d threads): %.6f seconds\n", num_threads, t1 - t0);
}


/*********************** REPORTING/PRINTING ***********************************/
void print_report(long long N, const Result *seqR, const Result *parR) {
    printf("\n=== Weather Data Analysis ===\n\n");
    printf("Number of Records: %lld\n", N);
    printf("Number of Cities: %d\n\n", NUM_CITIES);

    printf("Overall Maximum Temperature: %.2f °C\n", (double)parR->global_max);
    printf("Overall Minimum Temperature: %.2f °C\n\n", (double)parR->global_min);

    printf("Average Temperature Per City:\n");
    for (int c = 0; c < NUM_CITIES; ++c) {
        double avg = (parR->city_counts[c] ? parR->city_sums[c] / parR->city_counts[c] : 0.0);
        printf("City %d: %.2f °C\n", c + 1, avg);
    }
    printf("\nDays above 40°C: %lld\n\n", parR->hot_days);
}

/*********************** BENCHMARK SWEEP **************************************/
void benchmark(const int *city_ids, const float *temps, long long N) {
    FILE *csv = fopen("timings.csv", "w");
    if (!csv) die("cannot open timings.csv for writing");
    fprintf(csv, "threads,seq_s,par_s,speedup\n");

    // One sequential baseline
    Result seqR; sequential_analysis(city_ids, temps, N, &seqR);
    double t_seq = 0.0; // (captured from stdout is messy) — measure again, quietly
    {
        double t0 = omp_get_wtime();
        Result tmp; sequential_analysis(city_ids, temps, N, &tmp); // prints time
        double t1 = omp_get_wtime();
        t_seq = t1 - t0;
    }

    int maxT = omp_get_max_threads();
    for (int T = 1; T <= maxT; T *= 2) {
        Result parR;
        double t0 = omp_get_wtime();
        parallel_analysis(city_ids, temps, N, T, &parR); // prints time
        double t1 = omp_get_wtime();
        double t_par = t1 - t0;
        double speed = (t_par > 0 ? t_seq / t_par : 0.0);
        fprintf(csv, "%d,%.6f,%.6f,%.3f\n", T, t_seq, t_par, speed);
        fflush(csv);
    }

    fclose(csv);
    printf("\nWrote thread scaling results to timings.csv (import into Excel/Sheets to graph).\n");
}

/*********************** MAIN **************************************************/
int main(int argc, char **argv) {
    // Parse simple CLI
    int do_generate = 0, do_run = 0, do_bench = 0, threads = 4;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--generate")) do_generate = 1;
        else if (!strcmp(argv[i], "--run"))      do_run = 1;
        else if (!strcmp(argv[i], "--benchmark")) do_bench = 1;
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
            if (threads <= 0) threads = 1;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("Usage: %s [--generate] [--run --threads N] [--benchmark]\n", argv[0]);
            printf("Macros: -DNUM_CITIES=10 -DNUM_RECORDS=1000000 -DTEMP_MIN=10 -DTEMP_MAX=50\n");
            return 0;
        }
    }

    if (do_generate) {
        generate_input_file();
    }

    if (do_run || do_bench) {
        int *city_ids = NULL; float *temps = NULL;
        long long N = load_dataset(INPUT_FILE, &city_ids, &temps);

        Result seqR, parR;
        printf("Loaded %lld records from %s\n", N, INPUT_FILE);

        printf("\n--- Sequential Run ---\n");
        sequential_analysis(city_ids, temps, N, &seqR);

        printf("\n--- Parallel Run ---\n");
        parallel_analysis(city_ids, temps, N, threads, &parR);

        print_report(N, &seqR, &parR);

        // Print a headline speedup (using the times printed above would be nicer if captured; here we only indicate threads used)
        printf("(See console outputs for individual timings and speedup; use --benchmark for CSV.)\n");

        if (do_bench) {
            benchmark(city_ids, temps, N);
        }

        free(city_ids); free(temps);
    }

    if (!do_generate && !do_run && !do_bench) {
        printf("Nothing to do. Try: --generate then --run --threads 4 (or --benchmark)\n");
    }

    return 0;
}
