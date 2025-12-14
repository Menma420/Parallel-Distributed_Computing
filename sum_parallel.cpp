
#include "omp.h"
#include <iostream>


void partial_sum(int size, double* array, double* array1);

int main()
{
	int n = 1024;
	int thread_count = 8;
	double* array = new double[n];
	double* array1 = new double[thread_count];
	for(int i = 0; i < n; i++)
		array[i] = i;

#pragma omp parallel num_threads(thread_count)
{
	partial_sum(n, array, array1);

}

	double sum = 0;

	for(int i = 0; i < thread_count; i++)
		sum += array1[i];


	double formula_sum = (double)n*(n-1)/2;

	std::cout << "N: " << n << std::endl;
	std::cout << "Loop Sum:    " << sum << std::endl;
	std::cout << "Formula Sum: " << formula_sum << std::endl;

	delete[] array;
	return 0;
}

void partial_sum(int size, double* array, double* array1){
	int myrank = omp_get_thread_num();
	int thread_count=omp_get_num_threads();
	int mySize= size/thread_count;
	double mySum= 0;
	int myRange_low=mySize*myrank;
	int myRange_high=myRange_low+ mySize-1;
	for(int i = myRange_low; i <= myRange_high; i++)
		mySum += array[i];
	array1[myrank]= mySum;
}
