/******************************************
 * Benchmarking of baseline implementations of
 * GeMM.
 * Author: jfhansen
 * Last Modified: 15/07/2020
 *****************************************/
#include <benchmark/benchmark.h>
#include <iostream>
#include <cstddef>
#include <algorithm>
#include <random>
#include <thread>
#include <vector>
#include <assert.h>
//#include "serial_gemm.cpp"

#define BLOCKSIZE 16

void blocking_gemm(const double *A, const double *B, double *C, size_t N);
void chunked_gemm(const double *A, const double *B, double *C, size_t N);
void blocking_parallel_gemm(const double *A, const double *B, double *C,
    size_t N, size_t startRow, size_t endRow);

void print_matrix(const double *M, size_t N)
{
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
            std::cout << M[i*N + j] << " ";
        std::cout << std::endl;
    }
}

// Serial GeMM with matrix dimensions being powers of two.
static void BM_blocking_gemm_power_of_two(benchmark::State &s) {
    // Fetch matrix dimension
    size_t N = 1 << s.range(0);

    // Instantiate matrices
    double *A = new double[N*N];
    double *B = new double[N*N];
    double *C = new double[N*N];

    // Generate values from uniform distribution.
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<double> dist(-10,10);

    // Fill in A and B matrices with random values
    // Initialize C as zero-only matrix
    std::generate(A, A + N*N, [&] { return dist(rng); } );
    std::generate(B, B + N*N, [&] { return dist(rng); } );
    std::generate(C, C + N*N, [&] { return 0; } );

    // Perform main loop of benchmark
    for (auto _ : s)
        blocking_gemm(A, B, C, N);

    // Free up memory used
    delete [] A;
    delete [] B;
    delete [] C;
}

// Serial GeMM with matrix dimensions being powers of two.
static void BM_chunked_gemm_power_of_two(benchmark::State &s) {
    // Fetch matrix dimension
    size_t N = 1 << s.range(0);

    // Instantiate matrices
    double *A = new double[N*N];
    double *B = new double[N*N];
    double *C = new double[N*N];

    // Generate values from uniform distribution.
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<double> dist(-10,10);

    // Fill in A and B matrices with random values
    // Initialize C as zero-only matrix
    std::generate(A, A + N*N, [&] { return dist(rng); } );
    std::generate(B, B + N*N, [&] { return dist(rng); } );
    std::generate(C, C + N*N, [&] { return 0; } );

    // Perform main loop of benchmark
    for (auto _ : s)
        chunked_gemm(A, B, C, N);

    // Free up memory used
    delete [] A;
    delete [] B;
    delete [] C;
}

static void BM_blocking_parallel_gemm(benchmark::State &s) {
    // Fetch matrix dimensions
    size_t N = ( 1 << s.range(0) );

    // Instantiate matrices
    double *A = static_cast<double *>(aligned_alloc(64, N*N*sizeof(double)));
    double *B = static_cast<double *>(aligned_alloc(64, N*N*sizeof(double)));
    double *C = static_cast<double *>(aligned_alloc(64, N*N*sizeof(double)));

    // Generate matrix elements from uniform distribution
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<double> dist(-10,10);

    // Fill in A and B matrices with random values.
    // Initialize C as zero-only matrix
    std::generate(A, A+N*N, [&] { return dist(rng); });
    std::generate(B, B+N*N, [&] { return dist(rng); });
    std::generate(C, C+N*N, [&] { return dist(rng); });

    // Compute number of threads and preallocate memory for threads.
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Compute number of rows handled per thread
    size_t n_rows = (N+num_threads-1)/num_threads;
    assert(n_rows % BLOCKSIZE == 0);

    // Main benchmark loop
    size_t start_row, end_row;
    for (auto _ : s) {
        // Launch threads
        for (size_t i = 0; i < num_threads; i++) {
            start_row = i * n_rows;
            // Since number of elements may not be a multiple of num_threads
            // end_row is start_row+n_rows or N, whichever is smaller.
            end_row = std::min(start_row+n_rows, N);
            threads.emplace_back(blocking_parallel_gemm, A, B, C, N, start_row, end_row);
        }
        assert(end_row == N);
        // Wait for threads to finish
        for (auto &t : threads) t.join();
        // Clear threads for each iteration of benchmark.
        threads.clear();
    }
    // Free memory
    free(A);
    free(B);
    free(C);
}

//BENCHMARK(BM_serial_gemm)->DenseRange(8,10)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_blocking_gemm_power_of_two)->DenseRange(8,10)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_chunked_gemm_power_of_two)->DenseRange(8,10)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_blocking_parallel_gemm)->DenseRange(8,10)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
