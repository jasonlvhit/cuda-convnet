#ifndef HANDLER_CUH_
#define HANDLER_CUH_
#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define threadsPerBlock 1024
__constant__ int N;


__device__ void dot(float *a,
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
		atomicAdd(*c, cache[0]);
	}
}

/*
Weight format:
===============
| (x0, y0) | (x0, y1) | (x0, y2) | (x0, y3) |... |x0, yN) |
| (x1, y0) |...
...
| (xN, y0)

*/

__global__ void cnnFullyConnectedLayerForward(const float *input,
	const float *W, const float *b, float *output, int in_area,
	int in_depth, int out_area, int out_depth)
{

}

__global__ void cnnFullyConnectedLayerBackProp(){}

__global__ void cnnConvolutionalLayerForward(){}

__global__ void cnnConvolutionalLayerBackProp(){}

__global__ void cnnMaxpoolingLayerForward(){}

__global__ void cnnMaxpoolingLayerBackProp(){}

#endif