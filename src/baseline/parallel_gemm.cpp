/*******************************************
 * Baseline Parallel GeMM implementation
 * Author: jfhansen
 * Last modified: 15/07/2020
 ******************************************/

#include <cstddef>

void parallel_gemm(const double *A, const double *B, double *C, 
    size_t N, size_t startRow, size_t endRow)
{
    for (size_t i = startRow; i < endRow; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
}