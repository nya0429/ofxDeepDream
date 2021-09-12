#pragma once
#include "ofMain.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>

namespace ofxDeepDream {

	class OpticalFlowModule {

	public:
		OpticalFlowModule();
		void setup(cv::cuda::GpuMat& frame);
		void calc(cv::cuda::GpuMat& newframe8U);
		void remap(cv::cuda::GpuMat& dreamframe, cv::cuda::GpuMat& newframe, cv::cuda::GpuMat& prevframe);

		void calc(cv::cuda::GpuMat& dreamframe, cv::cuda::GpuMat& newframe, cv::cuda::GpuMat& prevframe);
		void calc(cv::cuda::GpuMat& dreamframe, cv::cuda::GpuMat& newframe, cv::cuda::GpuMat& prevframe, cv::cuda::GpuMat& prevGray);
		void blend(cv::cuda::GpuMat& dreamframe, cv::cuda::GpuMat& newframe, cv::cuda::GpuMat& prevframe);

		cv::cuda::GpuMat getOpticalflowView(cv::cuda::GpuMat& newframe, cv::cuda::GpuMat& prevframe);
		ofParameterGroup& getParameters() { return group; }

	private:

		ofParameterGroup group;
		ofParameter<double> weight1;
		//double weight1 = 0.01;

		cv::cuda::GpuMat prevGray;
		cv::cuda::GpuMat newGray;
		cv::cuda::GpuMat flow;
		cv::cuda::GpuMat planes[2];
		std::vector<cv::cuda::GpuMat> hsv;
		cv::cuda::GpuMat warpArange;

		//for debug
		cv::cuda::GpuMat magnitude;
		cv::cuda::GpuMat angle;
		cv::cuda::GpuMat saturation;
		cv::cuda::GpuMat opticalflow;

		//optiion
		cv::cuda::GpuMat norm_mag;
		cv::cuda::GpuMat flow_mask;

		cv::cuda::GpuMat background_blendimg;
		cv::cuda::GpuMat background_masked;

		cv::Ptr<cv::cuda::FarnebackOpticalFlow> opflow;

	};
}