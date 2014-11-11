#ifndef HANDLER_CUH_
#define HANDLER_CUH_
#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "math_functions.h"

#define threadsPerBlock 1024
__constant__ int out_area, in_area;

__global__ void cnnFullyConnectedLayerForward(const float *input,
	const float *W, const float *b, float *output) {
	__shared__ float cache[threadsPerBlock];
	int cacheIndex = threadIdx.x;

	if (cacheIndex < in_area)
		cache[cacheIndex] = input[cacheIndex] * W[blockIdx.x * in_area + cacheIndex];
	else
		cache[cacheIndex] = 0;
	
	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) {
		// sigmod
		output[blockIdx.x] = 1.0 / (1.0 + expf(cache[0] + b[blockIdx.x]));
	}
}


//void back_prop(){
//	/*
//	Compute the err terms;
//	*/
//	for (size_t in = 0; in < in_depth_; in++){
//		g_[in] = a_->df(input_[in]) * dot(this->next->g_, get_W_step(in));
//	}
//	/*
//	Update weights.
//	*/
//	for (size_t out = 0; out < out_depth_; out++){
//		for (size_t in = 0; in < in_depth_; in++){
//			auto delta = alpha_/*learning rate*/
//				* input_[in] * this->next->g_[out]/*err terms*/
//				/*lambda_ momentum*/
//				+ lambda_ * deltaW_[out * in_depth_ + in];
//			W_[out * in_depth_ + in] += delta;
//			/*update momentum*/
//			deltaW_[out * in_depth_ + in] = delta;
//		}
//		b_[out] += this->next->g_[out];
//	}
//}

__constant__ float alpha, lambda;

__global__ void cnnFullyConnectedLayerBackProp(const float* input, float* W, float* b,
	float* next_err_terms){
	__shared__ float cache[threadsPerBlock];
	if (threadIdx.x < in_area && blockIdx.x < out_area){
		W[threadIdx.x + blockIdx.x * in_area] += alpha *
			next_err_terms[blockIdx.x] * input[threadIdx.x];
	}
	b[blockIdx.x] += next_err_terms[blockIdx.x] / out_area;
}

__global__ void compute_err_terms(const float* input, const float* W, 
	const float* next_err_terms, float* err_terms){
	__shared__ float cache[threadsPerBlock];
	int cacheIndex = threadIdx.x;

	if (cacheIndex < out_area)
		cache[cacheIndex] = next_err_terms[cacheIndex] * W[blockIdx.x + in_area * cacheIndex];
	else
		cache[cacheIndex] = 0;

	__syncthreads();
	
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) {
		// sigmod derivative: f_x * (1.0 - f_x)
		err_terms[blockIdx.x] = input[blockIdx.x] * (1.0 - input[blockIdx.x]) * cache[0];
	}
}

__global__ void cnnConvolutionalLayerForward(){}

__global__ void cnnConvolutionalLayerBackProp(){}

__global__ void cnnMaxpoolingLayerForward(){}

__global__ void cnnMaxpoolingLayerBackProp(){}

#endif