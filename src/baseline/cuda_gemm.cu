/************************************
 * Baseline CUDA GeMM implementation
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
void cuda_gemm(const float *A, const float *B, float *C, size_t N)
{
	// Thread index across multiple blocks
	//const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t stride = blockDim.x;
	// Block index in grid
	const size_t col = blockIdx.x;
	// Thread index in block
	const size_t tid = threadIdx.x;

	// Shared memory: Each GPU block is responsible for dealing with 
	// the computation of at least one column of the C matrix.
	// Thereby a column of the B matrix can be brought into shared memory.
	// Since matrices are stored row-by-row, this saves a lot of time dealing
	// with cache misses.
	extern __shared__ float x[];
	// Every thread in block fetches one element in column 'bid'.
	for (size_t row = tid; row < N; row+=stride)
		x[row] = B[row*N+col];
	__syncthreads();
	
	// Perform GeMM
	for (size_t row = tid; row < N; row+=stride)
		for (size_t j = 0; j < N; j++)
			C[row*N+col] += A[row*N+j] * x[j];
	__syncthreads();
}

__global__
void count_zero_elems(const float *C, size_t N, float *nzelem)
{
	unsigned int stride = blockDim.x;
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Block loads elements into shared memory
	extern __shared__ float y[];
	// If element is zero write 1 in y array
	for (size_t row = tid; row < N; row+=stride)
		y[row] = (C[row*N+bid] == 0) ? 1.0 : 0.0;
	__syncthreads();
	
	// Build summation tree
	for (int s=N/2; s>0; s=s/2)
	{
		for (size_t row = tid; row < s; row+=stride)
		{
			if (row < s)
				y[row] += y[row+s];
			__syncthreads();
		}
	}
	// Thread 0 in every block holds amount of zero elements in one column.
	if (tid == 0)
		atomicAdd(nzelem, y[tid]);
	
	__syncthreads();
}

float verify_result(const float *A, const float *B, float *C, size_t N)
{
	unsigned count_v = 0;
	unsigned count_c = 0;
	float average_error = 0;
	std::cout << "Reached verify_result." << std::endl;
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
			average_error += (tmp - C[i*N+j]);
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
	int N = (1<<7);
	//int N = 10;
	uint32_t bytes = N*N*sizeof(float);	

	// Allocate memory in host.
	float *h_a, *h_b, *h_c;
	float *h_nzelem;
	h_a = new float[N*N];
	h_b = new float[N*N];
	h_c = new float[N*N];
	h_nzelem = new float;
	*h_nzelem = 0;
	
	// Allocate unified memory
	//cudaMallocManaged(&a, N*N*sizeof(float));
	//cudaMallocManaged(&b, N*N*sizeof(float));
	//cudaMallocManaged(&c, N*N*sizeof(float));

	// Generate values from uniform distribution
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_real_distribution<float> dist(-10,10);

	// Fill A and B matrices with random values
	// Initialize C as zero-only matrix
	std::generate(h_a, h_a+N*N, [&] { return 1.0; });//dist(rng); });
	std::generate(h_b, h_b+N*N, [&] { return 1.0; });//dist(rng); });
	std::generate(h_c, h_c+N*N, [&] { return 0.0; });

	// Allocate device memory
	float *d_a, *d_b, *d_c;
	float *d_nzelem;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);
	cudaMalloc(&d_nzelem, sizeof(float));
	
	// Copy data to device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nzelem, h_nzelem, sizeof(float), cudaMemcpyHostToDevice);
	
	// Compute number of threads per block and number of blocks
	unsigned blockSize = 256;
	unsigned numBlocks = N;
	
	count_zero_elems<<<numBlocks, blockSize, N>>>(d_c,N,d_nzelem);
	cudaDeviceSynchronize();

	cudaMemcpy(h_nzelem, d_nzelem, sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "Number of zero elements in C before: " << *h_nzelem << std::endl;
	*h_nzelem = 0;

	cudaMemcpy(d_nzelem, h_nzelem, sizeof(float), cudaMemcpyHostToDevice);
	
	// Run GeMM kernel on GPU
	cuda_gemm<<<numBlocks, blockSize, N>>>(d_a,d_b,d_c,N);
	// Wait for GPU to finish
	cudaDeviceSynchronize();
	
	count_zero_elems<<<numBlocks, blockSize, N>>>(d_c,N,d_nzelem);
	cudaDeviceSynchronize();
	
	cudaMemcpy(h_nzelem, d_nzelem, sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "Number of zero elements in C after: " << *h_nzelem << std::endl;
	
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
	//std::cout << "Result Matrix: " << std::endl;
	//for (int i = 0; i < N; i++) {
	//	for (int j = 0; j < N; j++)
	//		std::cout << h_c[i*N+j] << " ";
	//	std::cout << std::endl;
	//}
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
	
	return 0;
}
