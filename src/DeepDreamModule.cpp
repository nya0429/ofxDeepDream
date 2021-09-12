#include "ofMain.h"
#include "DeepDreamModule.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <math.h>


namespace F = torch::nn::functional;

namespace ofxDeepDream {

	DeepDreamModule::DeepDreamModule() {

		rng = cv::RNG();
		normalize.mean = normalize.mean.to(torch::Device(torch::kCUDA));
		normalize.stddev = normalize.stddev.to(torch::Device(torch::kCUDA));

		try {
			// Deserialize the ScriptModule from a file using torch::jit::load().
			ofFilePath  file;
			auto path = file.getAbsolutePath("../../../../../addons/ofxDeepDream/model/my_Inception_v3.pt");
			model = torch::jit::load(path, torch::kCUDA);
			//model = torch::jit::load("D:/Users/nagai/Documents/python/realtime-deepdream/my_Inception_v3.pt", torch::kCUDA);
			model.eval();
			model.train(false);
			for (auto& param : model.parameters()) {
				param.set_requires_grad(false);
			};
			std::cout << "finish load model" << std::endl;
		}
		catch (const c10::Error& e) {
			std::cerr << "error loading the model\n";
		}

		group.setName("DeepDreamModule");
		group.add(norm_str.set("norm_str", 5, 1, 8));
		group.add(lr.set("lr", 0.008, 0.005, 0.3));
		group.add(octave_scale.set("octave_scale", 1.2, 1.1, 1.7));

		group.add(global_octave_num.set("octave_num", 10, 1, 20));
		group.add(num_iterations.set("iteration", 1, 1, 4));

		group.add(limited_tensor_size.set("limited_tensor_size", 598, 299, 1024));

	}

	void DeepDreamModule::setup(cv::cuda::GpuMat& gpuMat) {

		int depth = gpuMat.depth();
		if (depth == CV_32F) {
			dataRange = 1.0;
			input_datatype = torch::kFloat32;
		}
		else if (depth == CV_8U) {
			dataRange = 255.0;
			input_datatype = torch::kUInt8;
		}
		else if (depth == CV_16F) {
			input_datatype = torch::kHalf;
		}

		if (gpuMat.channels() != 3) {
			std::cout << "unexpect channel num" << gpuMat.channels() << std::endl;
		}
	
	};

	torch::Tensor DeepDreamModule::preprocess(cv::cuda::GpuMat& gpuMat) {
	
		std::vector<int64_t> dims = { 1, gpuMat.channels(), gpuMat.rows, gpuMat.cols };
		int64 step = gpuMat.step1();
		std::vector<int64_t> strides = { 1, 1, step, gpuMat.channels() };
		auto options = torch::TensorOptions().dtype(input_datatype).device(torch::kCUDA);
		return torch::from_blob(gpuMat.data, dims, strides, options);
	};

	void DeepDreamModule::datacopy(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {

		std::vector<int64_t> dims = { 1, src.channels(), src.rows, src.cols };
		int64 step = src.step1();
		std::vector<int64_t> strides = { 1, 1, step, src.channels() };
		auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
		auto srcTensor = torch::from_blob(src.data, dims, strides, options);
		auto dstTensor = torch::from_blob(dst.data, dims, strides, options);
		srcTensor = srcTensor.contiguous();
		dstTensor = dstTensor.contiguous();
		dstTensor.data().copy_(srcTensor);

	};

	void DeepDreamModule::pipe_check(cv::cuda::GpuMat& gpuMat) {

		try {
			//torch::Tensor blob_tensor = preprocess(gpuMat);
			std::vector<int64_t> dims = { 1, gpuMat.channels(), gpuMat.rows, gpuMat.cols };
			int64 step = gpuMat.step1();
			std::vector<int64_t> strides = { 1, 1, step, gpuMat.channels() };
			auto options = torch::TensorOptions().dtype(input_datatype).device(torch::kCUDA);
			torch::Tensor blob_tensor = torch::from_blob(gpuMat.data, dims, strides, options);
			torch::Tensor tensor = blob_tensor.toType(tensor_datatype);
			//tensor = dreamer(tensor);
			////tensor = deprocess(tensor);
			tensor = tensor.squeeze().permute({ 1, 2, 0 });
			//tensor = tensor * _std + _mean;
			//tensor = torch::clamp(tensor * dataRange, 0, dataRange);
			tensor = tensor.contiguous();
			tensor = tensor.to(input_datatype);
			
			blob_tensor = blob_tensor.squeeze().permute({ 1, 2, 0 });
			blob_tensor.data().copy_(tensor);

		}
		catch (c10::Error e) { cout << e.what() << "\n"; }
		catch (cv::Exception e) { cout << e.what() << "\n"; }
		catch (std::runtime_error e) { cout << e.what() << "\n"; }

	}

	torch::Tensor DeepDreamModule::dreaming(torch::Tensor& tensor) {

		tensor = tensor.detach().set_requires_grad(true);

		for (int i = 0; i < num_iterations; i++) {

			std::vector<torch::jit::IValue> inputs;
			inputs.push_back(tensor);
			//inputs.push_back(c10::Scalar(1));
			auto out = model.forward(inputs).toTensor();
			torch::Tensor loss = out.norm();
			loss.backward();

			auto avg_grad = torch::abs(tensor.grad()).mean().item().toFloat();
			auto norm_lr = lr / avg_grad;

			tensor.set_data(tensor.tensor_data() + norm_lr * norm_str * tensor.grad().detach());
			torch::clamp(tensor, lower, upper);
			tensor.grad().detach().zero_();

			//del loss;

		};

		return tensor;
	}

	void DeepDreamModule::_dreamer(cv::cuda::GpuMat& gpuMat) {

		torch::Tensor blob_tensor = preprocess(gpuMat);
		torch::Tensor tensor = blob_tensor.toType(tensor_datatype);
		tensor = dreamer(tensor);
		tensor = deprocess(tensor);
		blob_tensor = blob_tensor.squeeze().permute({ 1, 2, 0 });
		blob_tensor.data().copy_(tensor);

	}

	cv::cuda::GpuMat DeepDreamModule::dreamer(cv::cuda::GpuMat& gpuMat) {

		torch::Tensor tensor = preprocess(gpuMat);
		tensor = tensor.toType(tensor_datatype);
		tensor = dreamer(tensor);
		tensor = deprocess(tensor);
		auto sizes = tensor.sizes();
		return cv::cuda::GpuMat(cv::Size(static_cast<int>(sizes[1]), static_cast<int>(sizes[0])), CV_32FC3, tensor.data_ptr());
	}
	
	torch::Tensor DeepDreamModule::scaling(torch::Tensor& x) {

		torch::Tensor x_org = x;
		torch::Tensor x_base;

		for (int ioct = 0; ioct < octave_num; ioct++) {

			auto size = x.sizes();
			int i = int(pow(octave_scale, (octave_num - ioct - 1)));
			int h = size[2]/i;
			int w = size[3]/i;
			x_base = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ h, w })).operator()(x_org);

			if (ioct == 0) {
				x = x_base;
			}
			else {
				x = x_base + F::interpolate(x, interpOpt.size(std::vector<int64_t>({ h,w })));
			}

			x = dreaming(x);
			x.set_data(x.data() - x_base);

		}

		x = x + x_base;
		return x;

	}

	torch::Tensor DeepDreamModule::loop_scaling(torch::Tensor& x) {

		torch::Tensor x_org = x;
		torch::Tensor x_base;
		auto org_size = x_org.sizes();
		auto size = x_org.sizes();

		//https://tzmi.hatenablog.com/entry/2020/03/10/230850
		//https://tzmi.hatenablog.com/entry/2019/12/30/220201

		double i = pow(octave_scale, (global_octave_num - global_octave_iterator - 1));
		//double i = pow(octave_scale, (global_octave_iterator));
		//i = pow(octave_scale, global_octave_num - 1) - i + 1.0;
		int h = limited_tensor_size / i;
		int w = limited_tensor_size / i;

		x_base = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ h, w })).operator()(x_org);
		x = x_base;
		x = dreaming(x);
		x.set_data(x.data() - x_base);

		x = x_org + F::interpolate(x, interpOpt.size(std::vector<int64_t>({ org_size[2], org_size[3] })));
		global_octave_iterator++;
		global_octave_iterator = global_octave_iterator % global_octave_num;

		return x;
	}

	torch::Tensor DeepDreamModule::dreamer(torch::Tensor& x) {
		//lr = random_lr[rng.uniform(0, 5)];
		//octave_scale = random_scale[rng.uniform(0, 5)];
		//x = x / dataRange;
		x = normalize.operator()(x);
		//x = scaling(x);
		x = loop_scaling(x);
		return x;

	}

	torch::Tensor DeepDreamModule::deprocess(torch::Tensor& tensor) {
		tensor = tensor.squeeze().permute({ 1, 2, 0 });
		tensor = tensor * _std + _mean;
		tensor = torch::clamp(tensor*dataRange, 0, dataRange).detach();
		tensor = tensor.to(input_datatype);
		return tensor.contiguous();
	}

}