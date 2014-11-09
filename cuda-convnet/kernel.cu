
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "data_descriptor.cuh"

#define threadsPerBlock 1024
__constant__ int N;


__global__ void dot(float *a,
	float *b, float *c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float   temp = 0;
	while (tid < N) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	// set the cache values
	cache[cacheIndex] = temp;

	// synchronize threads in this block
	__syncthreads();

	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) {
		atomicAdd(c, cache[0]);
	}
}

int main()
{
	int h_N = 10000;
	float *h_a, *h_b, *h_z;
	float *d_a, *d_b, *d_z;

	h_a = (float*)malloc(sizeof(float)*h_N);
	h_b = (float*)malloc(sizeof(float)*h_N);
	h_z = (float*)malloc(sizeof(float)*h_N);
	cudaMalloc((void**)&d_a, sizeof(float)*h_N);
	cudaMalloc((void**)&d_b, sizeof(float)*h_N);
	cudaMalloc((void**)&d_z, sizeof(float)*h_N);
	cudaMemcpyToSymbol(N, &h_N, sizeof(h_N));

	for (int i = 0; i < h_N; i++){
		h_a[i] = 1;
		h_b[i] = 1;
	}

	cudaMemcpy(d_a, h_a, sizeof(float) * h_N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(float) * h_N, cudaMemcpyHostToDevice);
	int num_block = h_N % threadsPerBlock ? h_N / threadsPerBlock + 1 : h_N / threadsPerBlock;
	dot <<< num_block, threadsPerBlock >>>(d_a, d_b, d_z);
	cudaDeviceSynchronize();
	cudaMemcpy(h_z, d_z, sizeof(float) * h_N, cudaMemcpyDeviceToHost);
	printf("%f\n", *h_z);
	getchar();
    return 0;
}
