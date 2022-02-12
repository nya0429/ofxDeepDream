#pragma once

#include "ofMain.h"
#include "DeepDreamModuleThread.h"
#include "SpynetOpticalFlowModule.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <atomic>

namespace ofxDeepDream {

    class ofxDeepDreamThread : public ofThread {

    public:
        ofxDeepDreamThread() {
            group.setName("ofxDeepDream");
            group.add(deepdream_thread.getParameters());
            group.add(blend_weight.set("blend_weight", 0.1, 0.0, 1.0));
        }

        ~ofxDeepDreamThread() {
            stop();
            waitForThread(false, 1000);
            ofLogNotice() << "destructor end";
        }

        void exit() {
            stop();
            waitForThread(false,1000);
            ofLogNotice(__FUNCTION__) << "Exit.";
            texShader.unload();
            deepdream_thread.exit();
        }

        void setup(ofTexture& tex1, ofTexture& tex2) {
            const ofTexture _tex1 = tex1;
            const ofTexture _tex2 = tex2;
            setup(_tex1, _tex2);
        }
        void setup(const ofTexture& tex1, const ofTexture& tex2) {

            ofTextureData texData = tex1.getTextureData();
            ofLogVerbose(__FUNCTION__) << "padding_w:"<< padding_w <<" padding_h:"<< padding_h;
            ofFboSettings s;
            s.width = texData.width - padding_w * 2;
            s.height = (texData.height - padding_h * 2) * 2;
            s.textureTarget = texData.textureTarget;
            s.internalformat = texData.glInternalFormat;
            s.textureTarget = GL_TEXTURE_2D; // Can't use GL_TEXTURE_RECTANGLE_ARB which is default in oF
            s.internalformat = GL_RGBA8;
            s.useDepth = true;
            s.useStencil = true;
            s.maxFilter = GL_LINEAR;
            s.minFilter = GL_LINEAR;
            s.numSamples = 4; // MSAA enabled. Anti-Alising is much important for VR experience

            mergeFbo.allocate(s);
            drawMergeFbo(tex1, tex2);

            s.textureTarget = texData.textureTarget;
            s.width = texData.width;
            s.height = texData.height;
            for (int i = 0; i < 2; i++) {
                eyeOutFbo[i].allocate(s);
            }

            setup(mergeFbo.getTexture());

        }
        void setup(ofTexture& tex) {
            const ofTexture _tex = tex;
            setup(_tex);
        }
        void setup(const ofTexture& tex) {

            try {
                inputTexData = tex.getTextureData();

                if (tex.getWidth()<299 || tex.getHeight() < 299) {
                    ofLogWarning(__FUNCTION__) << "unexpected size texture.";
                }

                GLenum format = ofGetGLFormatFromInternal(inputTexData.glInternalFormat);

                numChannel = format == GL_RGB ? 3 : 4;
                glGenBuffers(1, &pbo);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
                glBufferData(GL_PIXEL_UNPACK_BUFFER, tex.getWidth() * tex.getHeight() * numChannel, 0, GL_DYNAMIC_DRAW);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
                glBindTexture(inputTexData.textureTarget, inputTexData.textureID);
                glGetTexImage(inputTexData.textureTarget, 0, ofGetGLFormatFromInternal(inputTexData.glInternalFormat), ofGetGLTypeFromInternal(inputTexData.glInternalFormat), 0);
                glBindTexture(inputTexData.textureTarget, 0);
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                ofLogNotice(__FUNCTION__) << "texture width:" << inputTexData.width << " height:" << inputTexData.height << " numChannel:" << numChannel;

                pipeBuffer = cv::ogl::Buffer(inputTexData.height, inputTexData.width, CV_8UC(numChannel), pbo);
                m8U_inputBuffer = pipeBuffer.mapDevice();

                if (numChannel == 4) {
                    cv::cuda::cvtColor(m8U_inputBuffer, m8UC3_formatted, cv::COLOR_RGBA2RGB);
                    m8UC3_formatted.convertTo(m32FC3_dreamed, CV_32F, 1.0 / 255.0);
                }
                else {
                    m8UC3_formatted = m8U_inputBuffer;
                    m8U_inputBuffer.convertTo(m32FC3_dreamed, CV_32F, 1.0 / 255.0);
                }

                deepdream_thread.setup(m32FC3_dreamed);
                spynet_module.setup(m32FC3_dreamed);

                m32FC3_output = m32FC3_dreamed.clone();
                m32FC3_output_cache = m32FC3_dreamed.clone();
                m32FC3_dream_cache = m32FC3_dreamed.clone();
                m32FC3_input_cache = m32FC3_dreamed.clone();
                m32FC3_tmp = m32FC3_dreamed.clone();

                m8U_outputBuffer = m8U_inputBuffer.clone();
                pipeBuffer.unmapDevice();

                outputTexture.allocate(inputTexData.width, inputTexData.height, inputTexData.glInternalFormat,
                    inputTexData.textureTarget != GL_TEXTURE_2D
                );

                glGenBuffers(1, &output_pbo);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, output_pbo);
                glBufferData(GL_PIXEL_UNPACK_BUFFER, tex.getWidth() * tex.getHeight() * numChannel, 0, GL_DYNAMIC_DRAW);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                output_pipeBuffer = cv::ogl::Buffer(inputTexData.height, inputTexData.width, CV_8UC(numChannel), output_pbo);
                output_pipeBuffer.copyFrom(m8U_inputBuffer);

                outputTexData = outputTexture.getTextureData();
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, output_pbo);
                glBindTexture(outputTexData.textureTarget, outputTexData.textureID);
                glTexSubImage2D(outputTexData.textureTarget, 0, 0, 0, outputTexData.width, outputTexData.height, ofGetGLFormatFromInternal(outputTexData.glInternalFormat), ofGetGLTypeFromInternal(outputTexData.glInternalFormat), 0);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                glBindTexture(outputTexData.textureTarget, 0);
                
                start();

            }
            catch (cv::Exception& e)
            {
                const char* err_msg = e.what();
                std::cout << "exception caught: " << err_msg << std::endl;
                ofLogError(__FUNCTION__) << err_msg;
            }

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

        void reset() {

            if (isThreadRunning()) {
                ofLogNotice(__FUNCTION__) << "reset.";

                mutex.lock();
                upload_pbo();
                upload_data();
                m32FC3_input.copyTo(m32FC3_dreamed);
                m32FC3_input.copyTo(m32FC3_tmp);
                m32FC3_input.copyTo(m32FC3_input_cache);
                m32FC3_input.copyTo(m32FC3_output_cache);
                download_data();
                download_pbo();
                mutex.unlock();
            }
        }

        void pauseDeepDreamThread() {
            if (deepdream_thread.isThreadRunning()) {
                deepdream_thread.stop();
                ofLogNotice(__FUNCTION__) << "Deep dream thread is pause.";
            }
            else {
                ofLogWarning(__FUNCTION__) << "Deep dream thread is stop.";
            }
        }

        void resumeDeepDreamThread() {
            if (deepdream_thread.isThreadRunning()) {
                ofLogWarning(__FUNCTION__) << "Deep dream thread is running.";
            }
            else {
                reset();
                deepdream_thread.start();
                ofLogNotice(__FUNCTION__) << "Deep dream thread is resume.";
            }
        }

        void threadedFunction() {

            while (isThreadRunning()) {

                threadFrameNum++;

                mutex.lock();
                upload_data();
                mutex.unlock();

                if (deepdream_thread.isThreadRunning()) {
                    try {
                        bool isUpdate = deepdream_thread.update(m32FC3_output_cache, m32FC3_dreamed, m32FC3_dream_cache);
                        spynet_module.calc(m32FC3_input);
                        auto planes = spynet_module.get_flow();

                        cv::cuda::subtract(m32FC3_output_cache, m32FC3_input_cache, m32FC3_tmp);
                        cv::cuda::remap(m32FC3_tmp, m32FC3_tmp, planes[0], planes[1], cv::INTER_LINEAR, cv::BORDER_REFLECT101);
                        cv::cuda::add(m32FC3_tmp, m32FC3_input, m32FC3_tmp);

                        if (isUpdate) {
                            cv::cuda::addWeighted(m32FC3_dream_cache, blend_weight, m32FC3_dreamed, 1.0 - blend_weight, 0, m32FC3_dreamed);
                            cv::cuda::subtract(m32FC3_dreamed, m32FC3_dream_cache, m32FC3_dreamed);
                            cv::cuda::remap(m32FC3_dreamed, m32FC3_dreamed, planes[0], planes[1], cv::INTER_LINEAR, cv::BORDER_REFLECT101);
                            cv::cuda::add(m32FC3_tmp, m32FC3_dreamed, m32FC3_tmp);
                        }

                        m32FC3_input.copyTo(m32FC3_input_cache);
                        m32FC3_tmp.copyTo(m32FC3_output_cache);
                        m32FC3_tmp.copyTo(m32FC3_output);

                        cv::cuda::addWeighted(m32FC3_output, blend, m32FC3_input, 1.0 - blend, 0, m32FC3_output);
                    }
                    catch (c10::Error e) { ofLogError(__FUNCTION__) << e.what();}
                    catch (cv::Exception e) { ofLogError(__FUNCTION__) << e.what(); }
                    catch (std::runtime_error e) { ofLogError(__FUNCTION__) << e.what(); }
           
                }
                else {

                    m32FC3_input.copyTo(m32FC3_output);
                }
                m32FC3_output.convertTo(m32FC3_output, m32FC3_output.type(), 1.0 - blackout);
                mutex.lock();
                download_data();
                mutex.unlock();

            }
        }

        ofParameterGroup& getParameters() { return group; }
        ofParameterGroup& getDeepDreamGroup() { return deepdream_thread.getParameters(); }
        const auto& getDeepDreamThread() { return deepdream_thread; }

        ofTexture& getTexture() { return outputTexture; }
        std::array<ofFbo, 2>& getFboArray() { return eyeOutFbo; }
        int& getThreadFrameNum() { return threadFrameNum;}
        
        void update(ofTexture& tex1, ofTexture& tex2) {

            const ofTexture _tex1 = tex1;
            const ofTexture _tex2 = tex2;
            update(_tex1, _tex2);

        }
        void update(const ofTexture& tex1, const ofTexture& tex2) {

            drawMergeFbo(tex1, tex2);
            mergeFbo.getTexture();
            update();
            drawOutputFbo();
        }
        void update(ofTexture& tex) {

            const ofTexture _tex = tex;
            update(_tex);

        }
        void update(const ofTexture& tex) {
            update();
        }
        void update() {
            if (isThreadRunning()) {

                upload_pbo();
                download_pbo();

            }
        }

        void setLikeViveSRWorks(const ofTexture& tex) {

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

        void drawLikeViveSRWorks(int i) {

            ofDisableDepthTest();
            ofPushMatrix();
            ofMultMatrix(glm::scale(glm::vec3(1.f, 1.f, -1.f)));
            ofTranslate(0, 0, 2.f); // Translate image plane to far away

            texShader.begin();
            texShader.setUniformTexture("tex", eyeOutFbo[i].getTexture(), 0);
            renderRect.draw();
            texShader.end();

            ofPopMatrix();
            ofEnableDepthTest();

        }

        void drawLikeViveSRWorks(int i, glm::mat4 pose) {

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

        void setblend(float val) {
            blend = val;
        }
        void setblack(float val) {
            blackout = val;
        }
        void setpadding(float w, float h) { padding_w = w; padding_h = h; }

    protected:

        void drawMergeFbo(const ofTexture& tex1, const ofTexture& tex2) {

            ofDisableAlphaBlending();
            mergeFbo.begin();
            float half_h = mergeFbo.getHeight() / 2;
            tex1.drawSubsection(0, 0, mergeFbo.getWidth(), half_h, padding_w, padding_h, mergeFbo.getWidth(), half_h);
            tex2.drawSubsection(0, half_h, mergeFbo.getWidth(), half_h, padding_w, padding_h, mergeFbo.getWidth(), half_h);
            mergeFbo.end();
            ofEnableAlphaBlending();
        }
        void drawOutputFbo() {

            float half_h = outputTexture.getHeight() / 2;
            for (int i = 0; i < 2; i++) {
                eyeOutFbo[i].begin();
                //mergeFbo.getTexture().drawSubsection(padding_w, padding_h, mergeFbo.getWidth(), half_h, 0, i * half_h, mergeFbo.getWidth(), half_h);
                outputTexture.drawSubsection(padding_w, padding_h, outputTexture.getWidth(), half_h, 0, i * half_h, outputTexture.getWidth(), half_h);
                eyeOutFbo[i].end();
            }
        }
        void upload_pbo() {

            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
            glBindTexture(inputTexData.textureTarget, inputTexData.textureID);
            glGetTexImage(inputTexData.textureTarget, 0, ofGetGLFormatFromInternal(inputTexData.glInternalFormat), ofGetGLTypeFromInternal(inputTexData.glInternalFormat), 0);
            glBindTexture(inputTexData.textureTarget, 0);
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            m8U_inputBuffer = pipeBuffer.mapDevice();
            pipeBuffer.unmapDevice();

        }
        void download_pbo() {

            output_pipeBuffer.copyFrom(m8U_outputBuffer);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, output_pbo);
            glBindTexture(outputTexData.textureTarget, outputTexData.textureID);
            glTexSubImage2D(outputTexData.textureTarget, 0, 0, 0, outputTexData.width, outputTexData.height, ofGetGLFormatFromInternal(outputTexData.glInternalFormat), ofGetGLTypeFromInternal(outputTexData.glInternalFormat), 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glBindTexture(outputTexData.textureTarget, 0);

        }
        void upload_data() {

            if (numChannel == 4) {
                cv::cuda::cvtColor(m8U_inputBuffer, m8UC3_formatted, cv::COLOR_RGBA2RGB);
            }
            else {
                m8U_inputBuffer.copyTo(m8UC3_formatted);
            }
            m8UC3_formatted.convertTo(m32FC3_input, CV_32F, 1.0 / 255.0);

        }
        void download_data() {

            if (numChannel == 4) {
                m32FC3_output.convertTo(m8UC3_formatted, CV_8U, 255.0);
                cv::cuda::cvtColor(m8UC3_formatted, m8U_outputBuffer, cv::COLOR_RGB2RGBA);
            }
            else {
                m32FC3_output.convertTo(m8U_outputBuffer, CV_8U, 255.0);
            }

        }

        float padding_w = 80;
        float padding_h = 40;

        float blackout = 0.3;
        float blend = 0;

        ofMutex mutex;
        std::condition_variable condition;
        int threadFrameNum = 0;
        int numChannel = 3;

        ofParameterGroup group;
        ofParameter<float> blend_weight;

        SpynetOpticalFlowModule spynet_module;
        DeepDreamModuleThread deepdream_thread;

        //Buffer PipeLine
        GLuint pbo;
        cv::ogl::Buffer pipeBuffer;
        GLuint output_pbo;
        cv::ogl::Buffer output_pipeBuffer;

        //cv GpuMat
        cv::cuda::GpuMat m8U_inputBuffer;
        cv::cuda::GpuMat m8U_outputBuffer;
        cv::cuda::GpuMat m8UC3_formatted;

        cv::cuda::GpuMat m32FC3_dreamed;
        cv::cuda::GpuMat m32FC3_dream_cache;
        cv::cuda::GpuMat m32FC3_tmp;
        cv::cuda::GpuMat m32FC3_input;
        cv::cuda::GpuMat m32FC3_input_cache;
        cv::cuda::GpuMat m32FC3_output;
        cv::cuda::GpuMat m32FC3_output_cache;

        //Texture Data
        ofTextureData inputTexData;
        ofTextureData outputTexData;
        ofTexture outputTexture;
        ofFbo mergeFbo;

        //for VR
        std::array<ofFbo, 2> eyeOutFbo;

        //ViveSRWorks
        ofVboMesh renderRect;
        ofShader texShader;

    };
}