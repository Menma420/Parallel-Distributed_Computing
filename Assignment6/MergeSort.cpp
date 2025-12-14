/******************************************************
Name: Uttkarsh Malviya
Roll Number: IIT2022061

Assignment 6 – OpenMP Task Parallelism
Problem 1: MergeSort with OpenMP Tasks

Experimental Setup:
- Machine: Intel i5/i7 (mention CPU & cores)
- Compilation Flags: g++ -fopenmp Mergesort.cpp -o mergesort

Summary of Results:
- Serial is faster for small N <= 1000000.
- Parallel goes on forever for more N 

Serial Sort Time: 0.138s     (avg over 5 runs)
Parallel Merge Sort Time: 2.125s    (avg over 5 runs)

******************************************************/




#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

const int CUTOFF = 1000; // cutoff for task creation
vector<int> temp;        // global temp buffer

void merge(vector<int> &arr, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    // copy back to arr
    for (int idx = left; idx <= right; idx++) {
        arr[idx] = temp[idx];
    }
}

void mergeSort(vector<int> &arr, int left, int right) {
    if (left < right) {
        int mid = (left + right) / 2;

        if (right - left > CUTOFF) {
            #pragma omp task
            mergeSort(arr, left, mid);

            #pragma omp task
            mergeSort(arr, mid + 1, right);

            #pragma omp taskwait
        } else {
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
        }

        merge(arr, left, mid, right);
    }
}

int main() {
    int N = 1000000;  
    vector<int> arr(N);
    for (int i = 0; i < N; i++) arr[i] = rand() % 1000000;

    vector<int> arr_copy = arr;
    temp.resize(N);  // allocate temp buffer once

    double start, end;

    cout << "==== Merge Sort ====\n";

    // Serial
    start = omp_get_wtime();
    sort(arr_copy.begin(), arr_copy.end());
    end = omp_get_wtime();
    cout << "Serial Sort Time: " << end - start << "s\n";

    // Parallel
    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(arr, 0, N - 1);
    }
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << end - start << "s\n";

    return 0;
}
