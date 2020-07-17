/************************************
 * Baseline CUDA integer GeMM implementation
 * Author: jfhansen
 * Last modified: 15/07/2020
 ***********************************/

#include <iostream>
#include <cstddef>
#include <assert.h>
#include <algorithm>
#include <random>
#include <math.h>

#define BLOCKSIZE 256

// Kernel function that computes GeMM on CUDA threads
__global__
void cuda_gemm(const int *A, const int *B, int *C, size_t N)
{
	// Block index in grid
	const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
	// Thread index in block
	const size_t row = blockIdx.y * blockDim.y + threadIdx.y;

	// Perform GeMM
	for (size_t j = 0; j < N; j++)
		C[row*N+col] += A[row*N+j] * B[j*N+col];
}

__global__
void count_zero_elems(const int *C, size_t N, int *nzelem)
{
	unsigned int stride = blockDim.x;
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Block loads elements into shared memory
	extern __shared__ int y[];
	// If element is zero write 1 in y array
	// Each block takes care of 1 column of input array
	for (size_t row = tid; row < N; row+=stride)
		y[row] = (C[row*N+bid] == 0) ? 1 : 0;
	__syncthreads();
	
	// Build summation tree. Find amount of zero elements in each column
	for (int s=N/2; s>0; s=s/2)
	{
		for (size_t row = tid; row < s; row+=stride)
		{
			y[row] += y[row+s];
		}
	}
	__syncthreads();
	// Thread 0 in every block holds amount of zero elements in one column.
	if (tid == 0)
		atomicAdd(nzelem, y[tid]);
	
	__syncthreads();
}

float verify_result(const int *A, const int *B, int *C, size_t N)
{
	unsigned count_v = 0;
	unsigned count_c = 0;
	float average_error = 0;
	float max_error = 0;
	for (size_t i = 0; i < N; i++)
	{
        for (size_t j = 0; j < N; j++)
		{
			float tmp = 0;
            for (size_t k = 0; k < N; k++)
			{
                tmp += A[i*N + k] * B[k*N + j];
			}
			if (tmp == 0)
				count_v++;
			if (C[i*N+j] == 0)
				count_c++;
			average_error += fabs(tmp - C[i*N+j]);
			max_error = fmax(max_error, tmp - C[i*N+j]);
			//assert(tmp == C[i*N+j]);
		}
	}
	average_error /= N*N;
	std::cout << "Average Error: " << average_error << std::endl;
	std::cout << "Number of zero elements in verification: " << count_v << std::endl;
	std::cout << "Number of zero elements in C: " << count_c << std::endl;
	return max_error;
}

int main() {
	int N = (1<<10);
	uint32_t bytes = N*N*sizeof(int);	

	// Allocate memory in host.
	int *h_a, *h_b, *h_c;
	int *h_nzelems;
	h_a = new int[N*N];
	h_b = new int[N*N];
	h_c = new int[N*N];
	h_nzelems = new int;
	*h_nzelems = 0;
	
	// Allocate unified memory
	//cudaMallocManaged(&a, N*N*sizeof(float));
	//cudaMallocManaged(&b, N*N*sizeof(float));
	//cudaMallocManaged(&c, N*N*sizeof(float));

	// Generate values from uniform distribution
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_int_distribution<int> dist(-10,10);

	// Fill A and B matrices with random values
	// Initialize C as zero-only matrix
	std::generate(h_a, h_a+N*N, [&] { return dist(rng); });
	std::generate(h_b, h_b+N*N, [&] { return dist(rng); });
	std::generate(h_c, h_c+N*N, [&] { return 0.0; });

	// Allocate device memory
	int *d_a, *d_b, *d_c;
	int *d_nzelems;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);
	cudaMalloc(&d_nzelems, sizeof(int));
	
	// Copy data to device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nzelems, h_nzelems, sizeof(int), cudaMemcpyHostToDevice);

	// Compute block size and number of blocks for GeMM
	unsigned THREADS = sqrt(BLOCKSIZE);
	unsigned BLOCKS = N/THREADS;
	dim3 threads(THREADS,THREADS);
	dim3 blocks(BLOCKS,BLOCKS);
	// Run GeMM kernel on GPU
	cuda_gemm<<<blocks, threads>>>(d_a,d_b,d_c,N);
	// Wait for GPU to finish
	cudaDeviceSynchronize();
	
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	unsigned blockSize = BLOCKSIZE;
	unsigned numBlocks = N;
	count_zero_elems<<<numBlocks, blockSize, N*sizeof(int)>>>(d_c, N, d_nzelems);
	cudaDeviceSynchronize();

	cudaMemcpy(h_nzelems, d_nzelems, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << "Number of Zero elements in C: " << *h_nzelems << std::endl;

	// Verify result
	float maxError;
	maxError =	verify_result(h_a,h_b,h_c,N);
	std::cout << "Max Error: " << maxError << std::endl;
	
	// Free Memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	delete [] h_a;
	delete [] h_b;
	delete [] h_c;
	
	cudaError_t err;
	while ( (err = cudaGetLastError()) != cudaSuccess)
		std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
	return 0;
}
