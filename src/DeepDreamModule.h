#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

namespace F = torch::nn::functional;

namespace ofxDeepDream {
	class DeepDreamModule {

	public:
		DeepDreamModule();
		void setup(cv::cuda::GpuMat& gpuMat);
		torch::Tensor dreamer(torch::Tensor& tensor);
		cv::cuda::GpuMat dreamer(cv::cuda::GpuMat& gpuMat);
		void _dreamer(cv::cuda::GpuMat& gpuMat);
		void pipe_check(cv::cuda::GpuMat& gpuMat);
		ofParameterGroup& getParameters() { return group; }

	private:

		//DeepDream Parameter
		ofParameterGroup group;
		ofParameter<int> global_octave_num;
		ofParameter<int> num_iterations;

		ofParameter<float> octave_scale;
		ofParameter<float> lr;
		ofParameter<float> norm_str;

		ofParameter<int> limited_tensor_size;

		int octave_num = 1;
		//float octave_scale = 1.2;
		//int num_iterations = 1;
		//float lr = 0.008;

		cv::RNG rng;
		double random_lr[6] = { 0.01, 0.009, 0.008, 0.02, 0.03, 0.007 };
		double random_scale[6] = { 1.2, 1.4, 1.6, 1.8, 2.0, 2.2 };

		//Torch Parameter
		c10::ScalarType tensor_datatype = torch::kHalf;
		c10::ScalarType input_datatype = torch::kFloat32;
		float dataRange = 1.0;

		//static tensor
		torch::jit::script::Module model;
		const torch::TensorOptions options = torch::TensorOptions().dtype(tensor_datatype).device(torch::kCUDA);
		const torch::Tensor mean = torch::tensor({ 0.485, 0.456, 0.406 }, options);
		const torch::Tensor std = torch::tensor({ 0.229, 0.224, 0.225 }, options);
		const torch::Tensor upper = ((1 - mean) / std).reshape({ 1, -1, 1, 1 });
		const torch::Tensor lower = (-mean / std).reshape({ 1, -1, 1, 1 });
		const torch::Tensor _mean = mean.reshape({ 1, 1, 3 });
		const torch::Tensor _std = std.reshape({ 1, 1, 3 });
		torch::Tensor output;
		cv::cuda::GpuMat outputMat;
		torch::data::transforms::Normalize<> normalize = torch::data::transforms::Normalize<>({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 });
		F::InterpolateFuncOptions interpOpt = F::InterpolateFuncOptions().mode(torch::kBicubic).align_corners(false);



		//Function
		torch::Tensor dreaming(torch::Tensor& tensor);
		torch::Tensor scaling(torch::Tensor& tensor);
		int global_octave_iterator = 0;
		torch::Tensor loop_scaling(torch::Tensor& tensor);
		
		torch::Tensor preprocess(cv::cuda::GpuMat& gpuMat);
		torch::Tensor deprocess(torch::Tensor& tensor);


		//under develop
		void datacopy(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);

	};
}