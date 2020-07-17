/*******************************************
 * Parallel GeMM implementation using Blocking 
 *  Optimization
 * Author: jfhansen
 * Last modified: 15/07/2020
 ******************************************/

#include <cstddef>

#define BLOCKSIZE 16

void parallel_do_block(size_t N, size_t si, size_t sj, size_t sk,
    const double *A, const double *B, double *C)
{
    for (size_t i = si; i < si+BLOCKSIZE; i++)
        for (size_t j = sj; j < sj+BLOCKSIZE; j++)
        {
            double cij = C[i*N+j];
            for (size_t k = sk; k < sk+BLOCKSIZE; k++)
                cij += A[i*N+k] * B[k*N+j];
            C[i*N+j] = cij;
        }
}

void blocking_parallel_gemm(const double *A, const double *B, double *C, 
    size_t N, size_t startRow, size_t endRow)
{
    for (size_t si = startRow; si < endRow; si+=BLOCKSIZE)
        for (size_t sj = 0; sj < N; sj+=BLOCKSIZE)
            for (size_t sk = 0; sk < N; sk+=BLOCKSIZE)
                parallel_do_block(N, si, sj, sk, A, B, C);
}