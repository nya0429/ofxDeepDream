#include "ofApp.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


//--------------------------------------------------------------
void ofApp::setup(){

    ofSetLogLevel(OF_LOG_VERBOSE);
    ofLogToConsole();
    
    ofDisableArbTex();
    ofSetVerticalSync(false);
    ofSetFrameRate(0);

    panel.setup();
    panel.add(DeepDream.getParameters());

    int camWidth = 960;  // try to grab at this size.
    int camHeight = 540;
    //get back a list of devices.
    vector<ofVideoDevice> devices = vidGrabber.listDevices();

    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i].bAvailable) {
            //log the device
            ofLogNotice() << devices[i].id << ": " << devices[i].deviceName;
        }
        else {
            //log the device and note it as unavailable
            ofLogNotice() << devices[i].id << ": " << devices[i].deviceName << " - unavailable ";
        }
    }

    vidGrabber.setDeviceID(0);
    vidGrabber.setDesiredFrameRate(30);
    vidGrabber.initGrabber(camWidth, camHeight);

    while (true) {
        vidGrabber.update();
        if (vidGrabber.isFrameNew()) {
            DeepDream.setup(vidGrabber.getTexture());
            break;
        }
    }

}

//--------------------------------------------------------------
void ofApp::update() {
    vidGrabber.update();
    if (vidGrabber.isFrameNew()) {
        DeepDream.update();
    }
}

//--------------------------------------------------------------
void ofApp::draw() {

    ofDrawBitmapString("Press spacebar to pause/resume deepdream.", 20, 15);

    ofDisableAlphaBlending();
    DeepDream.getTexture().draw(20, 20);
    ofEnableAlphaBlending();
    panel.draw();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

    if (key == ' ') {
        if (DeepDream.getDeepDreamThread().isThreadRunning()) {
            DeepDream.pauseDeepDreamThread();
            ofLogNotice(__FUNCTION__) << "pause deepdream thread.";
        }
        else {
            DeepDream.resumeDeepDreamThread();
            ofLogNotice(__FUNCTION__) << "resume deepdream thread.";
        }
    }

    if (key == 17) {
        ofLogWarning(__FUNCTION__) << "Ctrl+Q : exit.";
        exit();
    }
}
//--------------------------------------------------------------