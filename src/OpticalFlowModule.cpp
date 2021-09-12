#include "OpticalFlowModule.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <numeric>
#include "Math.h"

namespace ofxDeepDream {

	OpticalFlowModule::OpticalFlowModule() {

		int numLevels = 2;
		double pyrScale = 0.5;
		bool fastPyramids = true;
		int winSize = 10;
		int numIters = 2;
		int polyN = 5;
		double polySigma = 1.2;
		int flags = 0;

		opflow = cv::cuda::FarnebackOpticalFlow::create(
			numLevels, pyrScale, fastPyramids, winSize, numIters, polyN, polySigma, flags
		);

		group.setName("OpticalFlowModule");
		group.add(weight1.set("blend_weight", 0.01, 0.0, 1.0));
		
	}

	template <typename T>
	cv::Mat createMat(T* data, int rows, int cols, int chs = 1) {
		// Create Mat from buffer 
		cv::Mat mat(rows, cols, CV_MAKETYPE(cv::DataType<T>::type, chs));
		memcpy(mat.data, data, rows * cols * chs * sizeof(T));
		return mat;
	}

	void OpticalFlowModule::setup(cv::cuda::GpuMat& frame) {
	
		cv::cuda::cvtColor(frame, prevGray, frame.channels() == 4 ? cv::COLOR_RGBA2GRAY : cv::COLOR_RGB2GRAY);

		int row = frame.size().height;
		int col = frame.size().width;

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

		warpArange.upload(m_merged);
		//std::cout << "exception caught: " << err_msg << std::endl;
		saturation = cv::cuda::GpuMat(frame.rows, frame.cols, CV_32FC1, (cv::Scalar)0);
		//std::cout << zero.size() << zero.channels()<< std::endl;

	};

	cv::cuda::GpuMat OpticalFlowModule::getOpticalflowView(cv::cuda::GpuMat& newframe, cv::cuda::GpuMat& prevframe) {
	//https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
	
		try {

			cv::cuda::cvtColor(newframe, newGray, cv::COLOR_RGB2GRAY);
			//newGray.convertTo(newGray, CV_8U, 255);
			opflow->calc(prevGray, newGray, flow);

			cv::cuda::cartToPolar(planes[0], planes[1], magnitude, angle,true);
			angle.convertTo(angle, angle.type(), 0.5);
			cv::cuda::normalize(magnitude, magnitude, double(0), double(255), cv::NORM_MINMAX, CV_32F);
			std::vector<cv::cuda::GpuMat> color_shuffle;
			color_shuffle.push_back(angle);
			color_shuffle.push_back(saturation);
			color_shuffle.push_back(magnitude);
			
			cv::cuda::merge(color_shuffle, opticalflow);
			opticalflow.convertTo(opticalflow, CV_8U);
			cv::cuda::cvtColor(opticalflow, opticalflow, cv::COLOR_HSV2RGB);
			opticalflow.convertTo(opticalflow, CV_32F, 1.0 / 255.0);

			prevGray.swap(newGray);

		}
		catch (cv::Exception& e)
		{
			const char* err_msg = e.what();
			std::cout << "exception caught: " << err_msg << std::endl;
		}

		return opticalflow;
	
	}

	void OpticalFlowModule::blend(cv::cuda::GpuMat& dreamframe, cv::cuda::GpuMat& newframe, cv::cuda::GpuMat& prevframe) {
		
		cv::cuda::addWeighted(newframe, weight1, dreamframe, 1.0-weight1, 0, dreamframe);
		cv::cuda::subtract(dreamframe, prevframe, dreamframe);
		cv::cuda::add(dreamframe, newframe, dreamframe);

	}

	void OpticalFlowModule::calc(cv::cuda::GpuMat& newframe8U) {

		//opflow calc
		cv::cuda::cvtColor(newframe8U, newGray, newframe8U.channels() == 4 ? cv::COLOR_RGBA2GRAY : cv::COLOR_RGB2GRAY);
		opflow->calc(prevGray, newGray, flow);
		flow.convertTo(flow, flow.type(), (double)-1);
		cv::cuda::add(flow, warpArange, flow);
		cv::cuda::split(flow, planes);
		prevGray.swap(newGray);

	};

	void OpticalFlowModule::remap(cv::cuda::GpuMat& dreamframe, cv::cuda::GpuMat& newframe, cv::cuda::GpuMat& prevframe) {
		cv::cuda::addWeighted(newframe, weight1, dreamframe, 1.0-weight1, 0, dreamframe);
		cv::cuda::subtract(dreamframe, prevframe, dreamframe);
		cv::cuda::remap(dreamframe, dreamframe, planes[0], planes[1], cv::INTER_LINEAR);
		cv::cuda::add(dreamframe, newframe, dreamframe);
	}

	void OpticalFlowModule::calc(cv::cuda::GpuMat& dreamframe, cv::cuda::GpuMat& newframe, cv::cuda::GpuMat& prevframe) {

		//opflow calc
		cv::cuda::cvtColor(newframe, newGray, cv::COLOR_RGB2GRAY);
		//newGray.convertTo(newGray, CV_8U, 255);
		opflow->calc(prevGray, newGray, flow);
		flow.convertTo(flow, flow.type(), (double)-1);
		cv::cuda::add(flow, warpArange, flow);
		cv::cuda::split(flow, planes);

		cv::cuda::addWeighted(newframe, weight1, dreamframe, 1.0 - weight1, 0.0, dreamframe);
		cv::cuda::subtract(dreamframe, prevframe, dreamframe);
		cv::cuda::remap(dreamframe, dreamframe, planes[0], planes[1], cv::INTER_LINEAR);
		cv::cuda::add(dreamframe, newframe, dreamframe);

		prevGray.swap(newGray);
	
	};


	void OpticalFlowModule::calc(cv::cuda::GpuMat& dreamframe, cv::cuda::GpuMat& newframe, cv::cuda::GpuMat& prevframe, cv::cuda::GpuMat& prevGray) {

		//opflow calc
		cv::cuda::cvtColor(newframe, newGray, cv::COLOR_RGB2GRAY);
		newGray.convertTo(newGray, CV_8U, 255);
		opflow->calc(prevGray, newGray, flow);
		flow.convertTo(flow, flow.type(), (double)-1);
		cv::cuda::add(flow, warpArange, flow);
		cv::cuda::split(flow, planes);
		cv::cuda::addWeighted(newframe, weight1, dreamframe, 1.0 - weight1, 0, dreamframe);
		cv::cuda::subtract(dreamframe, prevframe, dreamframe);
		cv::cuda::remap(dreamframe, dreamframe, planes[0], planes[1], cv::INTER_LINEAR);
		cv::cuda::add(dreamframe, newframe, dreamframe);
		prevGray.swap(newGray);

	};

}