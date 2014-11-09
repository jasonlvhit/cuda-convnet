#ifndef DATA_CUH_
#define DATA_CUH_

#include <vector>

struct cnnBaseDataDescriptor{
	size_t in_width;
	size_t in_height;
	size_t in_depth;
	size_t in_area;
	size_t out_width;
	size_t out_height;
	size_t out_depth;
	size_t out_area;

	std::vector<float_t> input;
	std::vector<float_t> W;
	std::vector<float_t> b;
	std::vector<float_t> output;

	float_t alpha;
	float_t lambda;

	cnnBaseDataDescriptor(size_t in_width_, size_t in_height_, size_t in_depth_,
		size_t out_width_, size_t out_height_, size_t out_depth_) :
		in_width(in_width_), in_height(in_height_), in_depth(in_depth_),
		out_height(out_height_), out_width(out_width_), out_depth(out_depth_),
		in_area(in_height_ * in_width_), out_area(out_height_ * out_width_)
	{}
};

struct cnnFullyConnectedLayerDataDescriptor :public cnnBaseDataDescriptor
{
	cnnFullyConnectedLayerDataDescriptor(size_t in_width_, size_t in_height_, size_t in_depth_, size_t out_width_,
		size_t out_height_, size_t out_depth_) :
		cnnBaseDataDescriptor(in_width_, in_height_, in_depth_, out_width_, out_height_, out_depth_)
	{}
};

struct cnnConvolutionalLayerDataDescriptor :public cnnBaseDataDescriptor
{
	cnnConvolutionalLayerDataDescriptor(size_t in_width_, size_t in_height_, size_t in_depth_,
		size_t out_depth_, size_t kernel_width_, size_t kernel_height_) :
		cnnBaseDataDescriptor(in_width_, in_height_, in_depth_, in_width - kernel_width_ + 1,
		in_height_ - kernel_height_ + 1, out_depth_),
		kernel_height(kernel_height_), kernel_width(kernel_height_),
		kernel_area(kernel_height * kernel_width)
	{}
	size_t kernel_width;
	size_t kernel_height;
	size_t kernel_area;
	std::vector<float_t> kernel;
};

struct cnnMaxPoolingLayerDataDescriptor :public cnnBaseDataDescriptor
{
	cnnMaxPoolingLayerDataDescriptor(size_t in_width_, size_t in_height_, size_t in_depth_) :
		cnnBaseDataDescriptor(in_width_, in_height_, in_depth_, in_width_ / 2, in_height_ / 2, in_depth_)
	{}
};

#endif // DATA_CUH_