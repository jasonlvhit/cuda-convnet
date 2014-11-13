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

__device__ float sigmod(float x){
	return 1.0 / (1.0 + expf(-x));
}

__device__ float df_sigmod(float x){
	return x * (1.0 - x);
}

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

//void forward(){
//	for (size_t out = 0; out < out_depth_; out++){
//		for (size_t in = 0; in < in_depth_; in++){
//			for (size_t h_ = 0; h_ < out_height_; h_++){
//				for (size_t w_ = 0; w_ < out_width_; w_++){
//					output_[getOutIndex(out, h_, w_)] +=
//						conv(getInforKernel(in, h_, w_), getW_(in, out));
//				}
//			}
//		}
//		for (size_t h_ = 0; h_ < out_height_; h_++){
//			for (size_t w_ = 0; w_ < out_width_; w_++){
//				output_[getOutIndex(out, h_, w_)] =
//					sigmod(output_[getOutIndex(out, h_, w_)] + /*eh?*/ b_[getb_(out, h_, w_)]);
//			}
//		}
//	}
//}

void __global__ ConvLayerSigmod(float* output, const float* b){
	int t_id = threadIdx.x;
	output[t_id] = 1.0 / (1.0 + expf(output[t_id] + b[t_id]));
}

void cnnConvolutionalLayerForward(const float* input, const float* W, 
	const float* b, float* output, size_t in_depth, size_t out_depth){
	for (size_t i = 0; i < in_depth; i++){
		for (size_t o = 0; o < out_depth; o++){
			dim3 grid();
			dim3 block();
			convolution << <grid, block >> >();
		}
	}
	cudaDeviceSynchronize();
	ConvLayerSigmod << < >> >();
}

__constant__ int kernel_width, in_width, in_heigh, out_width;

__global__ void convolution(const float* input, const float* W, float * output){
	int out_id = blockIdx.x + out_width * blockIdx.y;
	int cache_id = threadIdx.x + threadIdx.y * kernel_width;
	__shared__ float cache[32];
	if (out_id < out_area){
		cache[cache_id] = W[cache_id] *
			input[(blockIdx.y + threadIdx.y) * in_width + blockIdx.x + threadIdx.x];
	}
	for (int i = 25; i < 32; i++) cache[i] = 0;
	__syncthreads();

	int i = 16;
	cache_id = threadIdx.x + threadIdx.y * blockDim.x;
	while (i != 0) {
		if (cache_id < i)
			cache[cache_id] += cache[cache_id + i];
		__syncthreads();
		i /= 2;
	}
	if (cache_id == 0 && out_id < out_area) {
		output[out_id] += cache[0];
	}
}

//void back_prop(){
//	g_.clear();
//	g_.resize(in_width_ * in_height_ * in_depth_);
//	/*update err terms of this layer.*/
//	for (size_t out = 0; out < out_depth_; out++){
//		for (size_t in = 0; in < in_depth_; in++){
//			for (size_t w_ = 0; w_ < out_width_; w_++){
//				for (size_t h_ = 0; h_ < out_height_; h_++){
//					for (size_t y_ = 0; y_ < kernel_size_; y_++){
//						for (size_t x_ = 0; x_ < kernel_size_; x_++){
//							auto ff = in * in_width_ * in_height_ + (h_ + y_) *
//								in_width_ + (x_ + w_);
//							g_[ff] += /*next layer err terms*/
//								this->next->g_[out * out_width_ *
//								out_height_ + h_ * out_width_ + w_] *
//								/*weight*/
//								W_[in * out_depth_ * kernel_size_ * kernel_size_ +
//								out * kernel_size_ * kernel_size_ +
//								kernel_size_ * (kernel_size_ - y_ - 1) +
//								(kernel_size_ - 1 - x_)] *
//								/*df of input*/
//								df_sigmod(input_[ff]);
//						}
//					}
//				}
//			}
//		}
//	}
//
//	/*update weight*/
//	for (size_t out = 0; out < out_depth_; out++){
//		for (size_t in = 0; in < in_depth_; in++){
//			for (size_t h_ = 0; h_ < out_height_; h_++){
//				for (size_t w_ = 0; w_ < out_height_; w_++){
//					auto tt = getb_(out, h_, w_);
//					for (size_t y_ = 0; y_ < kernel_size_; y_++){
//						for (size_t x_ = 0; x_ < kernel_size_; x_++){
//							/*find update pixel*/
//							auto target = in * out_depth_ * kernel_size_ * kernel_size_ +
//								out * kernel_size_ * kernel_size_ +
//								kernel_size_ * (kernel_size_ - y_ - 1) +
//								(kernel_size_ - 1 - x_);
//							/*cal delta*/
//							auto delta = /*learning rate*/
//								alpha_ *
//								/*input*/
//								input_[in * in_width_ * in_height_ + (h_ + y_) *
//								in_width_ + (x_ + w_)] *
//								/*next layer err terms*/
//								this->next->g_[tt]
//								/*weight momentum*/
//								+ lambda_ * deltaW_[target];
//
//							W_[target] += delta;
//							/*update momentum*/
//							deltaW_[target] = delta;
//						}
//					}
//					b_[tt] += alpha_ * this->next->g_[tt];
//				}
//			}
//		}
//	}
//}

__global__ void cnnConvolutionalLayerBackProp(const float* input, const float* W,
const float* next_err_terms, float* err_terms, float* kernel, float* b)
{
	int e_id = (blockIdx.y + threadIdx.y) * in_width + blockIdx.x + threadIdx.x;
	int k_id = threadIdx.x + threadIdx.y * kernel_width;
	int o_id = blockIdx.y * out_width + blockIdx.x;
	/*err terms*/
	err_terms[e_id]
		+= kernel[k_id]
		*
		next_err_terms[o_id]
		* /*df sigmod*/
		input[e_id] * (1.0 - input[e_id]);
	/*update weights*/
	kernel[k_id] += alpha *
		next_err_terms[o_id]
		* input[e_id];
	b[o_id] += alpha * next_err_terms[o_id];
}

inline __device__ float f_max(const float l, const float r){
	return l > r ? l : r;
}

__global__ void cnnMaxpoolingLayerForward(const float* input, float* output, int* index)
{
	int out_x = threadIdx.x;
	int out_y = threadIdx.y;
	float lu = input[2 * out_x + 2 * out_y * in_width];
	float ru = input[2 * out_x + 2 * out_y * in_width + 1];
	float ld = input[2 * out_x + (2 * out_y + 1) * in_width];
	float rd = input[2 * out_x + 1 + (2 * out_y + 1) * in_width];
	
	int o_id = out_x + out_y * out_width;
	float max = lu;
	int i = 0;
	if (ru > max){
		max = ru, i = 1;
	}
	if (ld > max){
		max = ld, i = 2;
	}
	if (rd > max){
		max = rd, i = 3;
	}

	output[o_id] = max;
	index[o_id] = i;
}

__global__ void cnnMaxpoolingLayerBackProp(int* index, const float* next_err_terms, 
	float* err_terms){
	int out_x = threadIdx.x;
	int out_y = threadIdx.y;
	int o_id = out_x + out_y * out_width;
	if (index[o_id] == 0){
		err_terms[2 * out_x + 2 * out_y * in_width] = next_err_terms[o_id];
	}
	else if (index[o_id] == 1){
		err_terms[2 * out_x + 2 * out_y * in_width + 1] = next_err_terms[o_id];
	}
	else if (index[o_id] == 2){
		err_terms[2 * out_x + (2 * out_y + 1) * in_width] = next_err_terms[o_id];
	}
	else if (index[o_id] == 3){
		err_terms[2 * out_x + 1 + (2 * out_y + 1) * in_width] = next_err_terms[o_id];
	}
}

#endif