/*******************************************
 * Baseline Serial GeMM implementation
 * Author: jfhansen
 * Last modified: 15/07/2020
 ******************************************/

#include <cstddef>

void serial_gemm(const double *A, const double *B, double *C, size_t N)
{
    // Standard triple for-loop implementation of matrix multiplication
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < N; k++)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
}
