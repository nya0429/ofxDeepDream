#pragma once
#include "ofMain.h"
#include "OpticalFlowModule.h"
#include "DeepDreamModule.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/opengl.hpp>

namespace ofxDeepDream {

	class ofxDeepDream {

	public:
		
		ofxDeepDream();

		void setup(const ofTexture& tex1, const ofTexture& tex2);
		void setup(ofTexture& tex1, ofTexture& tex2);
		void setup(const ofTexture& tex);
		void setup(ofTexture& tex);

		void update(const ofTexture& tex1, const ofTexture& tex2);
		void update(ofTexture& tex1, ofTexture& tex2);
		void update(const ofTexture& tex);
		void update(ofTexture& tex);

		void reset();

		void setLikeViveSRWorks(const ofTexture& tex);
		void drawLikeViveSRWorks(int i, glm::mat4 pose);

		std::array<ofFbo, 2> getFboArray() { return eyeOutFbo; };
		ofTexture getTexture() { return outputTexture; };

		ofParameterGroup& getParameters() { return group; }

	private:

		ofParameterGroup group;
		ofParameter<bool> isEnable;

		DeepDreamModule deepdream;
		OpticalFlowModule opticalflow;

		//Buffer PipeLine
		GLuint pbo;
		cv::ogl::Buffer pipeBuffer;

		cv::cuda::GpuMat bufferMat;
		cv::cuda::GpuMat _BufferMat;
		cv::cuda::GpuMat _8UC3_BufferMat;
		cv::cuda::GpuMat _32FC3_BufferMat;
		cv::cuda::GpuMat dreamMat;
		cv::cuda::GpuMat cacheMat;

		ofTextureData inputTexData;
		ofTextureData outputTexData;
		ofTexture outputTexture;

		//for VR
		ofFbo mergeFbo;
		std::array<ofFbo, 2> eyeOutFbo;
		void drawOutputFbo();
		std::array<ofTexture, 2> eyeOutTexture;

		//ViveSRWorks
		ofVboMesh renderRect;
		ofShader texShader;

		//under development
		std::array<ofTexture, 2> getTextureArray();

		//for use texture2D
		void setupTexture2D(ofTexture& tex);
		void updateTexture2D(ofTexture& tex);
		void drawTexture2D();
		std::array<cv::ogl::Texture2D, 2> eyeInputTex;
		std::array<cv::ogl::Texture2D, 2> eyeOutputTex;
		std::array<cv::cuda::GpuMat, 2> eyeInputMat;
		std::array<cv::cuda::GpuMat, 2> eyeDreamMat;
		std::array<cv::cuda::GpuMat, 2> _eyeDreamMat;
		std::array<cv::cuda::GpuMat, 2> eyeCacheMat;
		std::array<cv::cuda::GpuMat, 2> grayCacheMat;

	};

}