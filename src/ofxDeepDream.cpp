#include "ofxDeepDream.h" 
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

namespace ofxDeepDream {

	ofxDeepDream::ofxDeepDream() {

		group.setName("ofxDeepDream");
		group.add(isEnable.set("isEnable", true));
		group.add(deepdream.getParameters());
		group.add(opticalflow.getParameters());

	}

	void ofxDeepDream::setup(ofTexture& tex1, ofTexture& tex2) {
		const ofTexture _tex1 = tex1;
		const ofTexture _tex2 = tex2;
		setup(_tex1, _tex2);
	}

	void ofxDeepDream::setup(const ofTexture& tex1, const ofTexture& tex2) {

		ofTextureData texData = tex1.getTextureData();
		ofFboSettings s;
		s.width = texData.width;
		s.height = texData.height * 2;
		s.textureTarget = texData.textureTarget;
		s.internalformat = texData.glInternalFormat;

		mergeFbo.allocate(s);
		mergeFbo.begin();
		tex1.draw(0, 0);
		tex2.draw(0, texData.height);
		mergeFbo.end();

		s.height = texData.height;
		for (int i = 0; i < 2; i++) {
			eyeOutFbo[i].allocate(s);
			eyeOutTexture[i].allocate(texData.width, texData.height, texData.glInternalFormat, false);
		}

		setup(mergeFbo.getTexture());
	
	}

	void ofxDeepDream::setup(ofTexture& tex) {
		const ofTexture _tex = tex;
		setup(_tex);
	}

	void ofxDeepDream::setup(const ofTexture& tex) {

		try {
			inputTexData = tex.getTextureData();
			GLenum format = ofGetGLFormatFromInternal(inputTexData.glInternalFormat);

			int channel = format == GL_RGB ? 3 : 4;
			glGenBuffers(1, &pbo);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, tex.getWidth() * tex.getHeight() * channel, 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
			glBindTexture(inputTexData.textureTarget, inputTexData.textureID);
			glGetTexImage(inputTexData.textureTarget, 0, ofGetGLFormatFromInternal(inputTexData.glInternalFormat), ofGetGLTypeFromInternal(inputTexData.glInternalFormat), 0);
			glBindTexture(inputTexData.textureTarget, 0);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

			std::cerr << "setTexture width:" << inputTexData.width << " height:" << inputTexData.height;
			std::cerr << "channel:" << channel;
			std::cerr << "\n";

			pipeBuffer = cv::ogl::Buffer(inputTexData.height, inputTexData.width, CV_8UC(channel), pbo);
			bufferMat = pipeBuffer.mapDevice();
			opticalflow.setup(bufferMat);

			if (channel == 4) {
				cv::cuda::cvtColor(bufferMat, _8UC3_BufferMat, cv::COLOR_RGBA2RGB);
				_8UC3_BufferMat.convertTo(dreamMat, CV_32F, 1.0 / 255.0);
			}else {
				bufferMat.convertTo(dreamMat, CV_32F, 1.0 / 255.0);
			}
			cacheMat = dreamMat.clone();
			pipeBuffer.unmapDevice();

			outputTexture.allocate(inputTexData.width, inputTexData.height, inputTexData.glInternalFormat,
				inputTexData.textureTarget != GL_TEXTURE_2D
			);

			outputTexData = outputTexture.getTextureData();
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
			glBindTexture(outputTexData.textureTarget, outputTexData.textureID);
			glTexSubImage2D(outputTexData.textureTarget, 0, 0, 0, outputTexData.width, outputTexData.height, ofGetGLFormatFromInternal(outputTexData.glInternalFormat), ofGetGLTypeFromInternal(outputTexData.glInternalFormat), 0);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			glBindTexture(outputTexData.textureTarget, 0);
		}
		catch (cv::Exception& e)
		{
			const char* err_msg = e.what();
			std::cout << "exception caught: " << err_msg << std::endl;
		}
	}

	void ofxDeepDream::update(ofTexture& tex1, ofTexture& tex2) {


		const ofTexture _tex1 = tex1;
		const ofTexture _tex2 = tex2;
		update(_tex1, _tex2);

	}

	void ofxDeepDream::update(const ofTexture& tex1, const ofTexture& tex2) {

		mergeFbo.begin();
		tex1.draw(0, 0);
		tex2.draw(0, tex1.getHeight());
		mergeFbo.end();

		update(mergeFbo.getTexture());

		drawOutputFbo();

	}

	void ofxDeepDream::update(ofTexture& tex) {

		const ofTexture _tex = tex;
		update(_tex);

	}

	void ofxDeepDream::update(const ofTexture& tex) {

		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
		glBindTexture(inputTexData.textureTarget, inputTexData.textureID);
		glGetTexImage(inputTexData.textureTarget, 0, ofGetGLFormatFromInternal(inputTexData.glInternalFormat), ofGetGLTypeFromInternal(inputTexData.glInternalFormat), 0);
		glBindTexture(inputTexData.textureTarget, 0);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

		if (isEnable) {
			try {

				deepdream._dreamer(dreamMat);
				bufferMat = pipeBuffer.mapDevice();
				opticalflow.calc(bufferMat);

				if (bufferMat.channels() == 4) {
					cv::cuda::cvtColor(bufferMat, _8UC3_BufferMat, cv::COLOR_RGBA2RGB);
					_8UC3_BufferMat.convertTo(_32FC3_BufferMat, CV_32F, 1.0 / 255.0);
				}
				else {
					bufferMat.convertTo(_32FC3_BufferMat, CV_32F, 1.0 / 255.0);
				}

				opticalflow.remap(dreamMat, _32FC3_BufferMat, cacheMat);
				_32FC3_BufferMat.swap(cacheMat);

				if (bufferMat.channels() == 4) {
					dreamMat.convertTo(_8UC3_BufferMat, CV_8U, 255.0);
					cv::cuda::cvtColor(_8UC3_BufferMat, _BufferMat, cv::COLOR_RGB2RGBA);
				}
				else {
					dreamMat.convertTo(_BufferMat, CV_8U, 255.0);
				}

				pipeBuffer.unmapDevice();
				pipeBuffer.copyFrom(_BufferMat);

			}
			catch (cv::Exception& e)
			{
				const char* err_msg = e.what();
				std::cout << "exception caught: " << err_msg << std::endl;
			}
		}

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(outputTexData.textureTarget, outputTexData.textureID);
		glTexSubImage2D(outputTexData.textureTarget, 0, 0, 0, outputTexData.width, outputTexData.height, ofGetGLFormatFromInternal(outputTexData.glInternalFormat), ofGetGLTypeFromInternal(outputTexData.glInternalFormat), 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glBindTexture(outputTexData.textureTarget, 0);
	}
	
	void ofxDeepDream::reset() {

		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
		glBindTexture(inputTexData.textureTarget, inputTexData.textureID);
		glGetTexImage(inputTexData.textureTarget, 0, ofGetGLFormatFromInternal(inputTexData.glInternalFormat), ofGetGLTypeFromInternal(inputTexData.glInternalFormat), 0);
		glBindTexture(inputTexData.textureTarget, 0);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

		try {

			bufferMat = pipeBuffer.mapDevice();

			if (bufferMat.channels() == 4) {
				cv::cuda::cvtColor(bufferMat, _8UC3_BufferMat, cv::COLOR_RGBA2RGB);
				_8UC3_BufferMat.convertTo(dreamMat, CV_32F, 1.0 / 255.0);
			}
			else {
				bufferMat.convertTo(dreamMat, CV_32F, 1.0 / 255.0);
			}
			dreamMat.copyTo(cacheMat);
			pipeBuffer.unmapDevice();

		}
		catch (cv::Exception& e)
		{
			const char* err_msg = e.what();
			std::cout << "exception caught: " << err_msg << std::endl;
		}

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(outputTexData.textureTarget, outputTexData.textureID);
		glTexSubImage2D(outputTexData.textureTarget, 0, 0, 0, outputTexData.width, outputTexData.height, ofGetGLFormatFromInternal(outputTexData.glInternalFormat), ofGetGLTypeFromInternal(outputTexData.glInternalFormat), 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glBindTexture(outputTexData.textureTarget, 0);
	
	}

	void ofxDeepDream::drawOutputFbo() {
	
		for (int i = 0; i < 2; i++) {
			ofTextureData texData = eyeOutFbo[i].getTexture().getTextureData();
			eyeOutFbo[i].begin();
			outputTexture.drawSubsection(0, 0, texData.width, texData.height, 0, i * texData.height, texData.width, texData.height);
			eyeOutFbo[i].end();
		}
	}

	void ofxDeepDream::setLikeViveSRWorks(const ofTexture& tex){
	
		std::string path = "../../../../../addons/ofxViveSRWorks/shader/";
		texShader.load(path + "texShader");

		glm::ivec2 distortedSize, undistortedSize;
		undistortedSize.x = tex.getWidth();
		undistortedSize.y = tex.getHeight();

		float aspect = float(undistortedSize.x) / float(undistortedSize.y);

		renderRect = ofMesh::plane(8.0, 8.0 / aspect, 2, 2);
		renderRect.clearTexCoords();
		renderRect.addTexCoord(glm::vec2(0, undistortedSize.y));
		renderRect.addTexCoord(glm::vec2(undistortedSize));
		renderRect.addTexCoord(glm::vec2(0));
		renderRect.addTexCoord(glm::vec2(undistortedSize.x, 0));

	}

	void ofxDeepDream::drawLikeViveSRWorks(int i, glm::mat4 pose) {

		ofDisableDepthTest();
		ofPushMatrix();
		ofMultMatrix(glm::scale(glm::vec3(1.f, 1.f, -1.f)) * pose);
		ofTranslate(0, 0, 2.f); // Translate image plane to far away

		texShader.begin();
		texShader.setUniformTexture("tex", eyeOutFbo[i].getTexture(), 0);
		renderRect.draw();
		texShader.end();

		ofPopMatrix();
		ofEnableDepthTest();

	}

	std::array<ofTexture, 2> ofxDeepDream::getTextureArray() {

		for (int i = 0; i < 1; i++) {
			ofTextureData texData = eyeOutTexture[i].getTextureData();
			//eyeOutFbo[i].begin();
			//outputTexture.drawSubsection(0, 0, texData.width, texData.height, 0, i * texData.height, texData.width, texData.height);
			//eyeOutFbo[i].end();
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
			glBindTexture(texData.textureTarget, texData.textureID);
			//PBO‚Ì“à—e‚ð•ÒW
			GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_WRITE);
			//if (ptr) {
			//	//for (int i = 0; i < 512 * 512; ++i) {
			//	//	GLubyte gray = (GLubyte)((ptr[3 * i] + ptr[3 * i + 1] + ptr[3 * i + 2]) / 3.0);
			//	//	ptr[3 * i] = gray;
			//	//	ptr[3 * i + 1] = gray;
			//	//	ptr[3 * i + 2] = gray;
			//	//}
			//}


			//for (int y = 0; y < sub_height; y++)
			//{
			//	char* row = m_data + ((y + sub_y) * img_width + sub_x) * 4;
			//	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, y, sub_width, 1, GL_RGBA, GL_UNSIGNED_BYTE, row);
			//}
			//auto* ptr = m_Pixels + (y * texData.width) * 4;
			//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, y, texData.width, texData.height, GL_RGBA, GL_UNSIGNED_BYTE, ptr);

			//glTexSubImage2D(texData.textureTarget, 0, 0, 0, texData.width, texData.height, ofGetGLFormatFromInternal(texData.glInternalFormat), ofGetGLTypeFromInternal(texData.glInternalFormat), ptr);
			//glUnmapBufferARB(GL_PIXEL_PACK_BUFFER);


			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			glBindTexture(texData.textureTarget, 0);

			eyeOutTexture[i].draw(0, 0);
		}
		return eyeOutTexture;
	}

	void ofxDeepDream::setupTexture2D(ofTexture& tex) {

		for (int i = 0; i < 1; i++) {

			auto texData = tex.getTextureData();
			eyeInputTex[i] = cv::ogl::Texture2D(
				texData.height,
				texData.width,
				texData.glInternalFormat == GL_RGB8 ? cv::ogl::Texture2D::RGB : cv::ogl::Texture2D::RGBA,
				(unsigned int)texData.textureID);

			eyeInputTex[i].copyTo(eyeInputMat[i]);
			eyeDreamMat[i] = eyeInputMat[i].clone();
			eyeCacheMat[i] = eyeInputMat[i].clone();
			eyeOutputTex[i].copyFrom(eyeInputMat[i]);

			eyeOutTexture[i].allocate(texData.width, texData.height, texData.glInternalFormat);
			eyeOutTexture[i].setUseExternalTextureID(eyeOutputTex[i].texId());
			
		}
		opticalflow.setup(eyeInputMat[0]);
	}
	void ofxDeepDream::updateTexture2D(ofTexture& tex) {

		eyeInputTex[0].copyTo(eyeInputMat[0]);
		deepdream.pipe_check(eyeInputMat[0]);
		//_eyeDreamMat[i] = deepdream.dreamer(eyeDreamMat[i]);
		//eyeDreamMat[i].swap(_eyeDreamMat[i]);
		//opticalflow.calc(eyeDreamMat[i], eyeInputMat[i], eyeCacheMat[i]);
		//eyeCacheMat[i].swap(eyeInputMat[i]);
		//eyeOutputTex[i].copyFrom(eyeDreamMat[i]);
		eyeOutputTex[0].copyFrom(eyeInputMat[0]);

		//for (int i = 0; i < 1; i++) {

		//}
	}
	void ofxDeepDream::drawTexture2D() {
		eyeOutTexture[0].draw(0,0);
	}

}