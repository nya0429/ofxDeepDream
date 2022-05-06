#pragma once
#include "ofMain.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <math.h>
#include <atomic>

namespace F = torch::nn::functional;

namespace ofxDeepDream {

	class DeepDreamModuleThread : public ofThread{

	public:

		DeepDreamModuleThread() {

			rng = cv::RNG();
			normalize.mean = normalize.mean.to(torch::Device(torch::kCUDA));
			normalize.stddev = normalize.stddev.to(torch::Device(torch::kCUDA));

			try {
				ofFilePath  file;
				auto path = file.getAbsolutePath("../../../../../addons/ofxDeepDream/model/Inception_v3.pt");
				ofLogNotice(__FUNCTION__) << "load model :" << path;
				model = torch::jit::load(path, torch::kCUDA);
				model.eval();
				model.train(false);
				for (auto& param : model.parameters()) {
					param.set_requires_grad(false);
				};
			}
			catch (const c10::Error& e) {
				ofLogError(__FUNCTION__) << e.msg() << "\n" << e.what();
			}

			group.setName("DeepDreamModule");
			group.add(layerLevel.set("layerlevel", init_layer, 0, 4));
			group.add(norm_str.set("norm_str", init_norm_str, 1, 4));
			group.add(lr.set("lr", init_lr, 0.005, 0.1));
			group.add(_octave_scale.set("octave_scale", init_oc_scale, 1.1, 1.7));
			group.add(_global_octave_num.set("octave_num", init_oc_num, 1, 20));
			group.add(_num_iterations.set("iteration", init_itr, 1, 20));
			group.add(straddle_scaleing.set("straddle_scaleing", true));
		}

		const float init_lr = 0.009;
		const float init_norm_str = 1.2;
		const float init_oc_scale = 1.2;
		const int init_oc_num = 20;
		const int init_itr = 20;
		const int init_layer = 4;

		~DeepDreamModuleThread() {
			stop();
			waitForThread(false,1000);
		}

		enum class status {
			wait_recieve = 0,
			processing = 1,
			wait_send = 2,
		};

		void setup(cv::cuda::GpuMat& gpuMat) {
			gpuMat.copyTo(outputGpuMat);
			gpuMat.copyTo(inputGpuMat);
			resultTensor = preprocess(gpuMat);
			resultTensor = resultTensor.squeeze().permute({ 1, 2, 0 });
			start();
		}

		void start() {
			startThread();
			ofLogNotice(__FUNCTION__) << "Thread start.";
		}
		void stop() {
			std::unique_lock<std::mutex> lck(mutex);
			stopThread();
			condition.notify_all();
			ofLogNotice(__FUNCTION__) << "Thread stop.";
		}
		void exit() {
			stop();
			waitForThread(false,1000);
			ofLogNotice(__FUNCTION__) << "Exit.";
		}
		void threadedFunction() {
			while (isThreadRunning()) {
				if (state == status::processing) {
					threadFrameNum++;
					std::unique_lock<std::mutex> lock(mutex);
					try {
						dreamer();
					}
					catch (c10::Error e) { ofLogError(__FUNCTION__) << e.what(); }
					catch (cv::Exception e) { ofLogError(__FUNCTION__) << e.what(); }
					catch (std::runtime_error e) { ofLogError(__FUNCTION__) << e.what(); }
					state = status::wait_send;
					condition.wait(lock);

				}
			}
		}
		bool update(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::GpuMat& cache) {

			try {

				if (state == status::wait_recieve) {

					std::unique_lock<std::mutex> lock(mutex);

					input.copyTo(inputGpuMat);
					inputTensor = preprocess(input);
					inputTensor = inputTensor.toType(tensor_datatype);
					state = status::processing;
					
					condition.notify_all();
					return false;
				}
				else if (state == status::wait_send) {
					
					std::unique_lock<std::mutex> lock(mutex);

					outputTensor = preprocess(output);
					outputTensor = outputTensor.squeeze().permute({ 1, 2, 0 });
					outputTensor.data().copy_(resultTensor);
					inputGpuMat.copyTo(cache);
					state = status::wait_recieve;

					condition.notify_all();
					return true;
				}
				else {
					return false;
				}

			}
			catch (c10::Error e) { ofLogError(__FUNCTION__) << e.what(); }
			catch (cv::Exception e) { ofLogError(__FUNCTION__) << e.what(); }
			catch (std::runtime_error e) { ofLogError(__FUNCTION__) << e.what(); }

		}
		ofParameterGroup& getParameters() { return group; }
		int getThreadFrameNum() {
			return threadFrameNum;
		}

		void dreamer(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output) {

			inputTensor = preprocess(input);
			inputTensor = inputTensor.toType(tensor_datatype);
			dreamer();
			torch::Tensor outputTensor = preprocess(output);
			outputTensor = outputTensor.squeeze().permute({ 1, 2, 0 });
			outputTensor.data().copy_(resultTensor);

		}

		void thread_dreamer(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output) {

			std::unique_lock<std::mutex> lock(mutex);

			input.copyTo(inputGpuMat);
			inputTensor = preprocess(inputGpuMat);
			inputTensor = inputTensor.toType(tensor_datatype);
			dreamer();
			torch::Tensor outputTensor = preprocess(output);
			outputTensor = outputTensor.squeeze().permute({ 1, 2, 0 });
			outputTensor.data().copy_(resultTensor);

			condition.notify_all();

		}

	protected:

		ofParameter<bool> straddle_scaleing;
		status state = status::wait_recieve;
		int threadFrameNum = 0;

		void dreamer() {

			resultTensor = inputTensor.clone();
			resultTensor = dreamer(inputTensor);
			resultTensor = deprocess(resultTensor);

		}

		torch::Tensor dreamer(torch::Tensor& x) {

			if (isRandomOctarveScale) {
				_global_octave_num = random_scale[rng.uniform(0, 5)];
			}

			global_octave_num = _global_octave_num;
			global_num_iterations = _num_iterations;
			if (global_octave_num < global_octave_iterator) {
				global_octave_iterator = 0;
			}
			if (global_num_iterations < global_num_iterator) {
				global_num_iterator = 0;
			}

			layerDepthLevel = layerLevel.get();
			octave_scale = _octave_scale;

			if (isRandomLr) {
				lr.set(random_lr[rng.uniform(0, 5)]);
			}

			x = normalize.operator()(x);

			try {
				if (straddle_scaleing) {
					x = loop_scaling(x);
				}
				else {
					x = scaling(x);
				}
				return x;
			}
			catch (const c10::Error& e) {
				ofLogError(__FUNCTION__) << e.msg() << "\n" << e.what();
			}
		}
		cv::cuda::GpuMat dreamer(cv::cuda::GpuMat& gpuMat) {

			torch::Tensor tensor = preprocess(gpuMat);
			tensor = tensor.toType(tensor_datatype);
			tensor = dreamer(tensor);
			tensor = deprocess(tensor);
			auto sizes = tensor.sizes();
			return cv::cuda::GpuMat(cv::Size(static_cast<int>(sizes[1]), static_cast<int>(sizes[0])), CV_32FC3, tensor.data_ptr());
		}
		void _dreamer(cv::cuda::GpuMat& gpuMat) {

			torch::Tensor blob_tensor = preprocess(gpuMat);
			torch::Tensor tensor = blob_tensor.toType(tensor_datatype);
			tensor = dreamer(tensor);
			tensor = deprocess(tensor);
			blob_tensor = blob_tensor.squeeze().permute({ 1, 2, 0 });
			blob_tensor.data().copy_(tensor);
		}

		torch::Tensor preprocess(cv::cuda::GpuMat& gpuMat) {
			std::vector<int64_t> dims = { 1, gpuMat.channels(), gpuMat.rows, gpuMat.cols };
			int64 step = gpuMat.step1();
			std::vector<int64_t> strides = { 1, 1, step, gpuMat.channels() };
			auto options = torch::TensorOptions().dtype(input_datatype).device(torch::kCUDA);
			return torch::from_blob(gpuMat.data, dims, strides, options);
		};

		torch::Tensor deprocess(torch::Tensor& tensor) {
			tensor = tensor.squeeze().permute({ 1, 2, 0 });
			tensor = tensor * _std + _mean;
			tensor = torch::clamp(tensor, 0.0, 1.0).detach();
			tensor = tensor.to(input_datatype);
			return tensor.contiguous();
		}

		torch::Tensor dreaming(torch::Tensor& tensor) {

			tensor = tensor.detach().set_requires_grad(true);

			for (int i = 0; i < global_num_iterations; i++) {
				
				std::vector<torch::jit::IValue> inputs;
				inputs.push_back(tensor);
				inputs.push_back(layerDepthLevel);

				auto out = model.forward(inputs).toTensor();
				torch::Tensor loss = out.norm();
				loss.backward();

				auto avg_grad = torch::abs(tensor.grad()).mean().item().toFloat();
				auto norm_lr = lr / avg_grad;

				tensor.set_data(tensor.tensor_data() + norm_lr * norm_str * tensor.grad().detach());
				torch::clamp(tensor, lower, upper);
				tensor.grad().detach().zero_();

			};
			return tensor;
		}

		torch::Tensor once_dreaming(torch::Tensor& tensor) {

			tensor = tensor.detach().set_requires_grad(true);

			std::vector<torch::jit::IValue> inputs;
			inputs.push_back(tensor);
			inputs.push_back(layerDepthLevel);
			auto out = model.forward(inputs).toTensor();
			torch::Tensor loss = out.norm();
			loss.backward();

			auto avg_grad = torch::abs(tensor.grad()).mean().item().toFloat();
			auto norm_lr = lr / avg_grad;

			tensor.set_data(tensor.tensor_data() + norm_lr * norm_str * tensor.grad().detach());
			torch::clamp(tensor, lower, upper);
			tensor.grad().detach().zero_();

			return tensor;
		}

		torch::Tensor scaling(torch::Tensor& x) {

			try {

				torch::Tensor x_org = x;
				torch::Tensor x_base;
				auto org_size = x_org.sizes();
				auto size = x_org.sizes();

				for (int ioct = 0; ioct < global_octave_num; ioct++) {

					int i = int(pow(octave_scale, (global_octave_num - ioct - 1)));
					int h = org_size[2] / i;
					int w = org_size[3] / i;
					h = max(299, h);
					w = max(299, w);
					x_base = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ h, w })).operator()(x_org);
				
					ofLogVerbose(__FUNCTION__, "octave: %d/%d", ioct+1, global_octave_num);

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
			catch (const c10::Error& e) {
				ofLogError(__FUNCTION__) << e.msg() << "\n" << e.what();
			}
		}

		torch::Tensor loop_scaling(torch::Tensor& x) {

			torch::Tensor x_org = x;
			torch::Tensor x_base;
			auto org_size = x_org.sizes();
			auto size = x_org.sizes();
			double i = pow(octave_scale, (global_octave_num - global_octave_iterator - 1));
			
			ofLogVerbose(__FUNCTION__, "octave: %d/%d", global_octave_iterator + 1, global_octave_num);
			ofLogVerbose(__FUNCTION__, "iter: %d/%d", global_num_iterator + 1, global_num_iterations);

			int h = org_size[2] / i;
			int w = org_size[3] / i;
			h = max(299, h);
			w = max(299, w);
			x_base = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ h, w })).operator()(x_org);
			x = x_base;
			x = once_dreaming(x);
			x.set_data(x.data() - x_base);
			x = x_org + F::interpolate(x, interpOpt.size(std::vector<int64_t>({ org_size[2], org_size[3] })));

			global_num_iterator++;

			if (global_num_iterator == global_num_iterations) {
				global_num_iterator = 0;
				global_octave_iterator++;
				global_octave_iterator = global_octave_iterator % global_octave_num;
			}
			return x;
		}

	private:
		torch::IValue layerDepthLevel = torch::IValue(0);
		ofParameter<int> layerLevel = 4;

		cv::RNG rng;
		ofMutex mutex;
		std::condition_variable condition;

		cv::cuda::GpuMat inputGpuMat;
		cv::cuda::GpuMat outputGpuMat;
		torch::Tensor inputTensor;
		torch::Tensor resultTensor;
		torch::Tensor outputTensor;

		ofParameterGroup group;

		//octave
		ofParameter<int> _global_octave_num;
		int global_octave_num;
		int global_octave_iterator = 0;
		
		//iterator
		int global_num_iterations;
		int global_num_iterator = 0;
		ofParameter<int> _num_iterations;

		//scale
		float octave_scale = 1.2;
		ofParameter<float> _octave_scale;
		ofParameter<bool> isRandomOctarveScale = false;
		double random_scale[6] = { 1.2, 1.4, 1.6, 1.8, 2.0, 2.2 };

		ofParameter<float> norm_str;
		ofParameter<float> lr;
		ofParameter<bool> isRandomLr = false;
		double random_lr[6] = { 0.01, 0.009, 0.008, 0.02, 0.03, 0.007 };

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

	};
}