/*******************************************
 * Serial GeMM implementation using Blocking
 *  Optimization
 * Author: jfhansen
 * Last modified: 15/07/2020
 ******************************************/

#include <cstddef>
//#include <x86intrin.h>

#define BLOCKSIZE 16

void chunked_gemm(const double *A, const double *B, double *C, size_t N)
{
    for (size_t row = 0; row < N; row++)
        for (size_t block = 0; block < N; block+=BLOCKSIZE)
            for (size_t chunk = 0; chunk < N; chunk+=BLOCKSIZE)
                for (size_t sub_chunk = 0; sub_chunk < BLOCKSIZE; sub_chunk++)
                    for (size_t idx = 0; idx < BLOCKSIZE; idx++)
                        C[row*N+block+idx] = A[row*N+chunk+sub_chunk] * B[(chunk+sub_chunk)*N+block+idx];
}
