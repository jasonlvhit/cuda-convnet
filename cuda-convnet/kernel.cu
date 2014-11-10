
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_descriptor.cuh"

#define threadsPerBlock 1024
__constant__ int out_area, in_area;

__device__ void dot(const float *a,
	const float *b, float *c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float   temp = 0;
	while (tid < in_area) {
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

/*
Weight format:
===============
| (x0, y0) | (x1, y0) | (x2, y0) | (x3, y0) |... |xN, y0) |
| (x1, y0) |...
...
| (xN, y0)

*/

__global__ void cnnFullyConnectedLayerForward(const float *input,
	const float *W, const float *b, float *output)
{
	for (int i = 0; i < out_area; i++){
		dot(input, W + in_area * i, output + i);
		output[i] = 1.0 / (1.0 + expf(-(output[i] + b[i])));
	}
}

int main()
{
	/*
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
	*/
	float *h_input, *h_output, *h_W, *h_b;
	float *d_input, *d_output, *d_W, *d_b;
	int h_in_area = 10000;
	int h_out_area = 10000;
	int d_in_area;
	int d_out_area;

	h_input = (float*)malloc(sizeof(float)*h_in_area);
	h_output = (float*)malloc(sizeof(float)*h_out_area);
	h_W = (float*)malloc(sizeof(float)*h_in_area *h_out_area);
	h_b = (float*)malloc(sizeof(float)*h_out_area);
	cudaMalloc((void**)&d_input, sizeof(float)*h_in_area);
	cudaMalloc((void**)&d_output, sizeof(float)*h_out_area);
	cudaMalloc((void**)&d_W, sizeof(float)*h_in_area *h_out_area);
	cudaMalloc((void**)&d_b, sizeof(float)*h_out_area);
	cudaMalloc((void**)&d_in_area, sizeof(int));
	cudaMalloc((void**)&d_out_area, sizeof(int));

	for (int i = 0; i < h_in_area; i++){
		h_input[i] = 1;
	}

	for (int i = 0; i < h_in_area * h_out_area; i++){
		h_W[i] = 1;
	}

	for (int i = 0; i < h_out_area; i++){
		h_output[i] = 0;
		h_b[i] = 1;
	}

	cudaMemcpy(d_input, h_input, sizeof(float)*h_in_area, cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, h_W, sizeof(float)*h_in_area*h_out_area, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(float)*h_out_area, cudaMemcpyHostToDevice);
	cudaMemcpy(&d_in_area, &h_in_area, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, sizeof(float)*h_out_area, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(in_area, &h_in_area, sizeof(h_in_area));
	cudaMemcpyToSymbol(out_area, &h_out_area, sizeof(h_out_area));
	int num_block = h_out_area % threadsPerBlock ? h_out_area / threadsPerBlock + 1 : h_out_area / threadsPerBlock;
	cnnFullyConnectedLayerForward << < num_block, threadsPerBlock >> >(d_input, d_W, d_b, d_output);
	cudaDeviceSynchronize();
	cudaMemcpy(h_output, d_output, sizeof(float)*h_out_area, cudaMemcpyDeviceToHost);

	for (int i = 0; i < h_out_area/10; i++){
		printf("%f\t", h_output[i]);
	}

	getchar();
    return 0;
}
