#pragma once
#include "ofMain.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include <math.h>

namespace F = torch::nn::functional;

namespace ofxDeepDream {

	class SpynetOpticalFlowModule {

	public:
		SpynetOpticalFlowModule() {
			try {
				ofFilePath  file;
				auto path = file.getAbsolutePath("../../../../../addons/ofxDeepDream/model/spynet.pt");
				ofLogNotice(__FUNCTION__) << "load model : " << path;

				spynet = torch::jit::load(path, torch::kCUDA);
				spynet.eval();
				spynet.train(false);
				for (auto& param : spynet.parameters()) {
					param.set_requires_grad(false);
				};
			}
			catch (const c10::Error& e) {
				ofLogError(__FUNCTION__) << e.what();
			}		
		
		}

		void setup(cv::cuda::GpuMat& gpuMat) {
			createWarpGrid(gpuMat.size());
			prevTensor = preprocess(gpuMat);
			c10::IntArrayRef size = prevTensor.sizes();
			int org_h = size[2];
			int org_w = size[3];
			int scale = pow(2, int(sizeFlag));
			int half_h = org_h / scale;
			int half_w = org_w / scale;
			int w = int(floor(ceil(half_w / 32.0) * 32.0));
			int h = int(floor(ceil(half_h / 32.0) * 32.0));
			PreprocessedSize = std::vector<int64_t>({ h,w });
			OriginalSize = std::vector<int64_t>({ org_h,org_w });
			prevTensor = F::interpolate(prevTensor, interpOpt.size(PreprocessedSize));
			float resizeWidth = float(org_w) / float(w);
			float resizeHeight = float(org_h) / float(h);
			resizeValue = std::vector<float>({ resizeWidth, resizeHeight });
			flow = cv::cuda::GpuMat(gpuMat.size(),CV_32FC2);
			outputTensor = preprocess(flow);
			ofLogNotice(__FUNCTION__, "spynet process size (%d,%d)", w, h);
		}

		auto& get_flow() {
			return planes;
		};

		void calc(cv::cuda::GpuMat& newframe) {
			auto tensor = preprocess(newframe);
			tensor = tensor.toType(tensor_datatype);
			tensor = F::interpolate(tensor, interpOpt.size(PreprocessedSize));
			//tensor = tensor.toType();
			auto result = estimate(tensor, prevTensor);
			resultTensor = deprocess(result);
			cv2_deprocess();
			prevTensor.data().copy_(tensor);
		}

		enum class size {
			full = 0,
			half = 1,
			quarter = 2,
		};

	protected:

		size sizeFlag = size::quarter;

		torch::Tensor preprocess(cv::cuda::GpuMat& gpuMat) {
			std::vector<int64_t> dims = { 1, gpuMat.channels(), gpuMat.rows, gpuMat.cols };
			int64 step = gpuMat.step1();
			std::vector<int64_t> strides = { 1, 1, step, gpuMat.channels() };
			auto options = torch::TensorOptions().dtype(input_datatype).device(torch::kCUDA);
			return torch::from_blob(gpuMat.data, dims, strides, options);
		};

		torch::Tensor estimate(torch::Tensor& newframe, torch::Tensor& prevframe) {

			newframe = newframe.detach_().set_requires_grad(false);
			prevframe = prevframe.detach_().set_requires_grad(false);
			std::vector<torch::jit::IValue> inputs;
			inputs.push_back(newframe);
			inputs.push_back(prevframe);
			auto result = spynet.forward(inputs).toTensor();
			return F::interpolate(result, interpOpt.size(OriginalSize));

		}

		torch::Tensor deprocess(torch::Tensor& tensor) {
			tensor = tensor.detach().to(input_datatype);
			return tensor.contiguous();
		}

		void cv2_deprocess() {
			//resultTensor = resultTensor * -1.0;
			outputTensor.data().copy_(resultTensor);
			//flow.convertTo(flow, flow.type(), (double)-1);
			cv::cuda::split(flow, planes);
			for (int i = 0; i < 2; i++) {
				planes[i].convertTo(planes[i], planes[i].type(), resizeValue[i]);
				cv::cuda::add(planes[i], warp[i], planes[i]);
			}
		}

		template <typename T>
		cv::Mat createMat(T* data, int rows, int cols, int chs = 1) {
			// Create Mat from buffer 
			cv::Mat mat(rows, cols, CV_MAKETYPE(cv::DataType<T>::type, chs));
			memcpy(mat.data, data, rows * cols * chs * sizeof(T));
			return mat;
		}

		void createWarpGrid(cv::Size& size) {

			int row = size.height;
			int col = size.width;

			std::vector<int> v(col);
			std::iota(v.begin(), v.end(), 0);

			cv::Mat m0 = cv::Mat::zeros(row, col, CV_32F);
			cv::Mat m1 = cv::Mat::zeros(row, col, CV_32F);
			cv::Mat row_mat = createMat<int>(v.data(), 1, col);

			for (int i = 0; i < row; i++) {
				m0.row(i) += row_mat;
				m1.row(i) += i;
			}

			std::vector<cv::Mat> mv;
			mv.push_back(m0);
			mv.push_back(m1);
			cv::Mat m_merged;
			cv::merge(mv, m_merged);

			warp[0].upload(m0);
			warp[1].upload(m1);

		}

		torch::jit::Module spynet;
		c10::ScalarType tensor_datatype = torch::kHalf;
		c10::ScalarType input_datatype = torch::kFloat32;
		F::InterpolateFuncOptions interpOpt = F::InterpolateFuncOptions().mode(torch::kBilinear).align_corners(false);
		std::vector<int64_t> PreprocessedSize;
		std::vector<int64_t> OriginalSize;
		std::vector<float> resizeValue;

		torch::Tensor prevTensor;
		torch::Tensor resultTensor;
		torch::Tensor outputTensor;
		cv::cuda::GpuMat flow;
		cv::cuda::GpuMat planes[2];
		cv::cuda::GpuMat warp[2];

	};
}