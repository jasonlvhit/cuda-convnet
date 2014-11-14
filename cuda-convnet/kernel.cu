
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "data_descriptor.cuh"
#include "mnist_parser.cuh"
using namespace std;

#define KERNEL_WIDTH 5
#define threadsPerBlock 1024
__constant__ int out_area, in_area;

__constant__ int kernel_width, in_width, in_height, out_width;

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

	int i =  16;
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


void train()
{
	vector<vector<float> > train_x;
	vector<float> train_y;
	LOAD_MNIST_TRAIN(train_x, train_y);

	float x[60000][28 * 28];
	float y[60000];

	for (int i = 0; i < train_x.size(); i++){
		for (int j = 0; j < train_x[i].size(); j++){
			x[i][j] = train_x[i][j];
		}
		y[i] = train_y[i];
	}

	float **d_x, *d_y;
	
	/*cnnConvolutionalLayerDataDescriptor *C1 = new cnnConvolutionalLayerDataDescriptor(28, 28, 1, 6, 5, 5);
	cnnMaxPoolingLayerDataDescriptor *S2 = new cnnMaxPoolingLayerDataDescriptor(24, 24, 6);
	cnnConvolutionalLayerDataDescriptor *C3 = new cnnConvolutionalLayerDataDescriptor(12, 12, 6, 100, 12, 12);
	cnnFullyConnectedLayerDataDescriptor *F4 = new cnnFullyConnectedLayerDataDescriptor(100, 10);*/



	cudaMalloc((void**)d_x, sizeof(float) * 60000 * 28 * 28);
	cudaMalloc((void**)d_y, sizeof(float) * 60000);

	float* d_C1_input, *d_C1_output, *d_C1_W, *d_C1_b, *d_C1_err_terms;
	float *d_S2_input, *d_S2_output, *d_S2_err_terms;
	float *d_C3_input, *d_C3_output, *d_C3_W, *d_C3_b, *d_C3_err_terms;
	float *d_F4_input, *d_F4_output, *d_F4_W, *d_F4_b, *d_F4_err_terms;
	float *d_O_input, *d_O_err_terms;

	cudaMalloc((void**)d_C1_input, sizeof(float) * 28 * 28);
	cudaMalloc((void**)d_C1_output, sizeof(float) * 24 * 24 * 6);
	cudaMalloc((void**)d_C1_W, sizeof(float) * 5 * 5 * 6);
	cudaMalloc((void**)d_C1_b, sizeof(float) * 24 * 24 * 6);
	cudaMalloc((void**)d_C1_err_terms, sizeof(float) * 28 * 28);

	cudaMalloc((void**)d_S2_input, sizeof(float) * 24 * 24 * 6);
	cudaMalloc((void**)d_S2_output, sizeof(float) * 12 * 12 * 6);
	cudaMalloc((void**)d_S2_err_terms, sizeof(float) * 24 * 24 * 6);

	cudaMalloc((void**)d_C3_input, sizeof(float) * 12 * 12 * 6);
	cudaMalloc((void**)d_C3_output, sizeof(float) * 1 * 1 * 100);
	cudaMalloc((void**)d_C3_W, sizeof(float) * 12 * 12 * 6 * 100);
	cudaMalloc((void**)d_C3_b, sizeof(float)* 1 * 1 * 100);
	cudaMalloc((void**)d_C3_err_terms, sizeof(float) * 12 * 12 * 6);
	
	cudaMalloc((void**)d_F4_input, sizeof(float) * 100);
	cudaMalloc((void**)d_F4_output, sizeof(float) * 10);
	cudaMalloc((void**)d_F4_W, sizeof(float) * 100 * 10);
	cudaMalloc((void**)d_F4_b, sizeof(float) * 10);
	cudaMalloc((void**)d_F4_err_terms, sizeof(float) * 100);

	cudaMalloc((void**)d_O_err_terms, sizeof(float) * 10);
	cudaMalloc((void**)d_O_input, sizeof(float) * 10);

	cudaMemcpy(d_x, x, sizeof(float) * 60000 * 28 * 28, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, sizeof(float) * 60000, cudaMemcpyHostToDevice);

	

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
	
	float *h_input, *h_output, *h_W, *h_b;
	float *d_input, *d_output, *d_W, *d_b;
	int h_in_area = 1000;
	int h_out_area = 1024;
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
	cnnFullyConnectedLayerForward << < h_out_area, threadsPerBlock >> >(d_input, d_W, d_b, d_output);
	cudaDeviceSynchronize();
	cudaMemcpy(h_output, d_output, sizeof(float)*h_out_area, cudaMemcpyDeviceToHost);

	for (int i = 0; i < h_out_area; i++){
		printf("%f\t", h_output[i]);
	}
	*/

	float* h_input, *h_output, *h_W;
	int h_in_width = 16;
	int h_in_height = 16;
	int h_in_area = h_in_height * h_in_width;
	int h_kernel_width = 5;
	int h_kernel_height = 5;
	int h_out_width = h_in_width - h_kernel_width + 1;
	int h_out_height = h_in_height - h_kernel_height + 1;
	int h_out_area = h_out_width * h_out_height;
	h_input = (float*)malloc(sizeof(float)*h_in_area);
	h_output = (float*)malloc(sizeof(float)*h_out_area);
	h_W = (float*)malloc(sizeof(float)*h_kernel_width * h_kernel_height);

	for (int i = 0; i < h_in_area; i++){
		h_input[i] = 1.0;
	}

	for (int i = 0; i < h_out_area; i++){
		h_output[i] = 0.0;
	}

	for (int i = 0; i < h_kernel_height * h_kernel_width; i++){
		h_W[i] = 1.0;
	}

	float* d_input, *d_output, *d_W;
	cudaMalloc((void**)&d_input, sizeof(float)*h_in_area);
	cudaMalloc((void**)&d_output, sizeof(float)*h_out_area);
	cudaMalloc((void**)&d_W, sizeof(float)*h_kernel_width * h_kernel_height);
	cudaMemcpy(d_input, h_input, sizeof(float)*h_in_area, cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, h_W, sizeof(float)*h_kernel_width*h_kernel_width, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, sizeof(float)*h_out_area, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(in_area, &h_in_area, sizeof(h_in_area));
	cudaMemcpyToSymbol(out_area, &h_out_area, sizeof(h_out_area));
	cudaMemcpyToSymbol(in_width, &h_in_width, sizeof(h_in_width));
	cudaMemcpyToSymbol(out_width, &h_out_width, sizeof(h_out_width));
	cudaMemcpyToSymbol(kernel_width, &h_kernel_width, sizeof(h_kernel_height));
	cudaMemcpyToSymbol(in_height, &h_in_height, sizeof(h_in_height));
	dim3 grid(h_out_width, h_out_height);
	dim3 block(8, 8);
	convolution<<< grid, block >>>(d_input, d_W, d_output);
	cudaDeviceSynchronize();
	cudaMemcpy(h_output, d_output, sizeof(float)*h_out_area, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < h_out_area; i++){
		printf("%f\t", h_output[i]);
	}
	
	
	getchar();
    return 0;
}
