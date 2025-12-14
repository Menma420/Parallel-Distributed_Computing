/*
 * Sparse Matrix (COO/MatrixMarket) × Dense Vector using CSR and pthreads
 * -----------------------------------------------------------------------
 * Name      : Uttkarsh Malviya
 * Roll No.  : IIT2022061
 *
 * Experimental setup: Intel(R) Core(TM) i5-10500H CPU @ 2.50GHz
 *
 * Summary:
 *  - Reads a Matrix Market file (coordinate real [symmetric|general])
 *  - Converts COO → CSR (mirrors off-diagonals if symmetric)
 *  - Reads dense vector from text (one number per line or whitespace-separated)
 *  - Computes SpMV sequentially (C1) and in parallel via pthreads (C2)
 *  - Verifies correctness (||C1 - C2||∞)
 *  - Benchmarks for threads in {1,2,4,8,16,32} and prints a timing table
 *
 * Usage:
 *  $ g++ -O3 -march=native -pthread Assignment3.cpp -o spmm
 *  $ ./spmm /path/to/inputfile.mtx /path/to/vector.txt [threads]
 *
 * Notes:
 *  - If [threads] is omitted, defaults to 4 for the C2 printout and also sweeps the
 *    benchmark set {1,2,4,8,16,32} bounded by #rows.
 *  - Prints matrix metadata and CSR arrays (row_ptr, col_idx, data) as required.
 * 
 *      #Rows: 138
        #Cols: 138
        Matrix is symmetric (mirrored off-diagonals if symmetric).

        Timing comparison (ms):
        Threads   Time_ms        Speedup_vs_1
        1         0.258          0.02
        2         0.421          0.01
        4         0.608          0.01
        8         1.065          0.00
        16        1.042          0.00
        32        1.514          0.00


        Conclusion: No real performance gain of parallel over sequential as the example vector is small. 
        Thus, Parallelism becomes beneficial only for much larger sparse matrices with high non-zero counts.
 * 
 */

#include <bits/stdc++.h>
#include <pthread.h>
using namespace std;

struct CSR {
    int nrows = 0, ncols = 0;               // dimensions
    vector<size_t> row_ptr;                 // size nrows+1
    vector<int> col_idx;                    // size nnz_full
    vector<double> data;                    // size nnz_full
};

struct COOTriplet { int r, c; double v; };

struct ThreadArgs {
    const CSR* A;               // CSR matrix
    const double* x;            // input vector
    double* y;                  // output vector
    int r0, r1;                 // row range [r0, r1)
};

static void* spmv_worker(void* arg) {
    ThreadArgs* t = reinterpret_cast<ThreadArgs*>(arg);
    const CSR& A = *t->A;
    const double* x = t->x;
    double* y = t->y;
    for (int r = t->r0; r < t->r1; ++r) {
        double sum = 0.0;
        size_t start = A.row_ptr[r];
        size_t end   = A.row_ptr[r+1];
        for (size_t k = start; k < end; ++k) {
            sum += A.data[k] * x[A.col_idx[k]];
        }
        y[r] = sum;
    }
    return nullptr;
}

// Basic Matrix Market header parser: returns (nrows, ncols, nnz, is_symmetric)
static tuple<int,int,long long,bool> parse_mm_header(istream& in) {
    string line;
    bool symmetric = false;
    // First non-empty line must start with %%MatrixMarket
    while (getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '%') {
            // Header line contains type
            if (line.rfind("%%MatrixMarket", 0) == 0) {
                // parse tokens
                // Example: %%MatrixMarket matrix coordinate real symmetric
                stringstream ss(line);
                string mm, obj, fmt, field, symmetry;
                ss >> mm >> obj >> fmt >> field >> symmetry;
                if (!symmetry.empty()) {
                    // lowercase
                    for (auto& ch : symmetry) ch = tolower(ch);
                    if (symmetry == "symmetric") symmetric = true;
                }
            }
            continue;
        } else {
            // This should be the size line: nrows ncols nnz
            int nrows=0, ncols=0; long long nnz=0;
            stringstream ss(line);
            ss >> nrows >> ncols >> nnz;
            if (!ss.fail()) return {nrows, ncols, nnz, symmetric};
        }
    }
    throw runtime_error("Invalid Matrix Market header");
}

static vector<COOTriplet> read_coo_entries(istream& in, long long nnz) {
    vector<COOTriplet> T;
    T.reserve((size_t)nnz);
    string line;
    while ((long long)T.size() < nnz && getline(in, line)) {
        if (line.empty() || line[0] == '%') continue;
        stringstream ss(line);
        int i, j; double v; ss >> i >> j >> v;
        if (ss.fail()) continue;
        // Convert to 0-based
        T.push_back({i-1, j-1, v});
    }
    if ((long long)T.size() != nnz) {
        throw runtime_error("Expected nnz entries but found fewer lines.");
    }
    return T;
}

static CSR coo_to_csr(int nrows, int ncols, const vector<COOTriplet>& coo, bool symmetric) {
    // If symmetric: mirror off-diagonals
    vector<COOTriplet> work;
    work.reserve(coo.size() * (symmetric ? 2ULL : 1ULL));
    for (const auto& t : coo) {
        work.push_back(t);
        if (symmetric && t.r != t.c) work.push_back({t.c, t.r, t.v});
    }
    // Sort by (row, col) for CSR
    sort(work.begin(), work.end(), [](const COOTriplet& a, const COOTriplet& b){
        if (a.r != b.r) return a.r < b.r; return a.c < b.c;
    });

    CSR A; A.nrows = nrows; A.ncols = ncols;
    A.row_ptr.assign(nrows + 1, 0);
    A.col_idx.reserve(work.size());
    A.data.reserve(work.size());

    int cur_row = 0;
    size_t idx = 0;
    while (cur_row < nrows && idx < work.size()) {
        // Advance row_ptr to the first entry for cur_row
        while (cur_row < work[idx].r) {
            A.row_ptr[cur_row+1] = A.col_idx.size();
            ++cur_row;
        }
        // Aggregate duplicates in the same (r,c) if present
        int r = work[idx].r;
        int c = work[idx].c;
        double sum = 0.0;
        while (idx < work.size() && work[idx].r == r && work[idx].c == c) {
            sum += work[idx].v;
            ++idx;
        }
        A.col_idx.push_back(c);
        A.data.push_back(sum);
        // Now continue for this row until it changes
        while (idx < work.size() && work[idx].r == r) {
            int cc = work[idx].c; double vv = work[idx].v; ++idx;
            // Combine consecutive duplicates if any (already sorted)
            while (idx < work.size() && work[idx].r == r && work[idx].c == cc) {
                vv += work[idx].v; ++idx;
            }
            A.col_idx.push_back(cc);
            A.data.push_back(vv);
        }
        A.row_ptr[r+1] = A.col_idx.size();
        cur_row = r+1;
    }
    // Finalize remaining empty rows
    while (cur_row < nrows) {
        A.row_ptr[cur_row+1] = A.col_idx.size();
        ++cur_row;
    }
    return A;
}

static vector<double> read_vector_file(const string& path, int expected_n) {
    ifstream fin(path);
    if (!fin) throw runtime_error("Failed to open vector file: " + path);
    vector<double> v; v.reserve(expected_n);
    double x;
    while (fin >> x) v.push_back(x);
    if ((int)v.size() != expected_n) {
        throw runtime_error("Vector length (" + to_string(v.size()) + ") does not match matrix rows (" + to_string(expected_n) + ")");
    }
    return v;
}

static vector<double> spmv_seq(const CSR& A, const vector<double>& x) {
    vector<double> y(A.nrows, 0.0);
    for (int r = 0; r < A.nrows; ++r) {
        double sum = 0.0;
        for (size_t k = A.row_ptr[r]; k < A.row_ptr[r+1]; ++k) {
            sum += A.data[k] * x[A.col_idx[k]];
        }
        y[r] = sum;
    }
    return y;
}

static vector<double> spmv_parallel(const CSR& A, const vector<double>& x, int nthreads) {
    nthreads = max(1, min(nthreads, A.nrows));
    vector<double> y(A.nrows, 0.0);
    vector<pthread_t> threads(nthreads);
    vector<ThreadArgs> args(nthreads);

    int rows_per_thread = (A.nrows + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        int r0 = t * rows_per_thread;
        int r1 = min(A.nrows, r0 + rows_per_thread);
        args[t] = ThreadArgs{&A, x.data(), y.data(), r0, r1};
        pthread_create(&threads[t], nullptr, spmv_worker, &args[t]);
    }
    for (int t = 0; t < nthreads; ++t) pthread_join(threads[t], nullptr);
    return y;
}

static double max_abs_diff(const vector<double>& a, const vector<double>& b) {
    double m = 0.0; size_t n = a.size();
    for (size_t i = 0; i < n; ++i) m = max(m, fabs(a[i] - b[i]));
    return m;
}

static void print_vector(const vector<double>& v, const string& name) {
    cout << name << " (n=" << v.size() << "):\n";
    for (size_t i = 0; i < v.size(); ++i) {
        cout << fixed << setprecision(6) << v[i];
        if (i + 1 != v.size()) cout << ' ';
    }
    cout << "\n\n";
}

static void print_csr(const CSR& A) {
    cout << "CSR Row Ptr (size=" << A.row_ptr.size() << "):\n";
    for (size_t i = 0; i < A.row_ptr.size(); ++i) {
        cout << A.row_ptr[i] << (i+1==A.row_ptr.size()? '\n' : ' ');
    }
    cout << "\nCSR Col Indices (nnz=" << A.col_idx.size() << "):\n";
    for (size_t i = 0; i < A.col_idx.size(); ++i) {
        cout << A.col_idx[i] << (i+1==A.col_idx.size()? '\n' : ' ');
    }
    cout << "\nCSR Data (nnz=" << A.data.size() << "):\n";
    cout << fixed << setprecision(10);
    for (size_t i = 0; i < A.data.size(); ++i) {
        cout << A.data[i] << (i+1==A.data.size()? '\n' : ' ');
    }
    cout << "\n";
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <matrix.mtx> <vector.txt> [threads]\n";
        cerr << "Example: ./spmm /mnt/data/inputfile.mtx /mnt/data/vector.txt 8\n";
        return 1;
    }

    const string mtx_path = argv[1];
    const string vec_path = argv[2];
    int threads_for_C2 = (argc >= 4 ? max(1, atoi(argv[3])) : 4);

    ifstream fin(mtx_path);
    if (!fin) {
        cerr << "Failed to open matrix file: " << mtx_path << "\n";
        return 1;
    }

    // Parse header & size
    int nrows, ncols; long long nnz; bool is_sym;
    tie(nrows, ncols, nnz, is_sym) = parse_mm_header(fin);

    // Read COO entries
    auto coo = read_coo_entries(fin, nnz);
    fin.close();

    // Convert to CSR (mirror if symmetric)
    CSR A = coo_to_csr(nrows, ncols, coo, is_sym);

    // Read vector
    auto x = read_vector_file(vec_path, A.nrows);

    // Metadata + matrix print
    cout << "#Rows: " << A.nrows << "\n";
    cout << "#Cols: " << A.ncols << "\n";
    cout << "#Non-Zeroes (after CSR build): " << A.data.size() << "\n";
    cout << "#Threads (C2 print): " << threads_for_C2 << "\n";
    cout << "Matrix is " << (is_sym ? "symmetric" : "general") << " (mirrored off-diagonals if symmetric).\n\n";

    print_csr(A);

    // Sequential
    auto t0 = chrono::high_resolution_clock::now();
    auto C1 = spmv_seq(A, x);
    auto t1 = chrono::high_resolution_clock::now();
    double seq_ms = chrono::duration<double, milli>(t1 - t0).count();

    // Parallel C2 (for given thread count)
    t0 = chrono::high_resolution_clock::now();
    auto C2 = spmv_parallel(A, x, threads_for_C2);
    t1 = chrono::high_resolution_clock::now();
    double par_ms = chrono::duration<double, milli>(t1 - t0).count();

    // Print results
    print_vector(C1, "C1 (sequential)");
    print_vector(C2, "C2 (parallel, threads=" + to_string(threads_for_C2) + ")");

    double err = max_abs_diff(C1, C2);
    cout << "Max |C1 - C2| = " << scientific << setprecision(6) << err << "\n";
    cout.unsetf(std::ios::floatfield);

    // Benchmark sweep
    vector<int> thread_sweep = {1,2,4,8,16,32};
    cout << "\nTiming comparison (ms):\n";
    cout << left << setw(10) << "Threads" << setw(15) << "Time_ms" << "Speedup_vs_1" << "\n";

    // Time seq once more for fair baseline with same warm caches? We'll use earlier seq_ms.
    // Measure for each thread count (cap by nrows)
    double t1_ms_baseline = seq_ms; // baseline ~ threads=1 sequential

    for (int t : thread_sweep) {
        int used_t = max(1, min(t, A.nrows));
        auto tA0 = chrono::high_resolution_clock::now();
        auto y = spmv_parallel(A, x, used_t);
        auto tA1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(tA1 - tA0).count();
        double speedup = t1_ms_baseline / ms;
        cout << left << setw(10) << used_t << setw(15) << fixed << setprecision(3) << ms << fixed << setprecision(2) << speedup << "\n";
    }

    return 0;
}
