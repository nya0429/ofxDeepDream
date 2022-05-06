#include "ofApp.h"
#include <stdio.h>

void ofApp::allocateTexture() {

	eyeWidth = vr.getHmd().getEyeWidth();
	eyeHeight = vr.getHmd().getEyeHeight();

	eyeWidth_eighth = vr.getHmd().getEyeWidth() / 8.f;
	eyeWidth_eighth = 480;

	eyeHeight_eighth = vr.getHmd().getEyeHeight() / 8.f;
	eyeHeight_eighth = 480;


	// Let's prepare fbos to be renderred for each eye
	ofFboSettings s;
	s.width = eyeWidth;   // Actual rendering resolution is higer than display(HMD) spec has,
	s.height = eyeHeight; // because VR system will distort images like barrel distortion and fit them through lens
	s.maxFilter = GL_LINEAR;
	s.minFilter = GL_LINEAR;
	s.numSamples = 4; // MSAA enabled. Anti-Alising is much important for VR experience
	s.textureTarget = GL_TEXTURE_2D; // Can't use GL_TEXTURE_RECTANGLE_ARB which is default in oF
	s.internalformat = GL_RGBA8;
	s.useDepth = true;
	s.useStencil = true;

	ofBackground(0);

	for (auto& f : eyeFbo) {
		f.allocate(s);
	}

}

void ofApp::setup() {

	ofSetLogLevel(OF_LOG_VERBOSE);
	ofLogToConsole();

	ofSetVerticalSync(false);
	vr.setup();
	vive.init(false);

	allocateTexture();

	panel.setup();
	panel.setPosition(760,0);
	panel.add(vive.getParameters());
	panel.add(DeepDream.getParameters());
	panel.add(isDrawVrView.set("isDrawVrView", true));
	panel.minimizeAll();
	
	vr.update();
	vive.update();
	DeepDream.setup(vive.getUndistortedTexture(0), vive.getUndistortedTexture(1));
	DeepDream.setLikeViveSRWorks(vive.getUndistortedTexture(0));

}

void ofApp::update(){

	vr.update();
	vive.update();

	DeepDream.update(vive.getUndistortedTexture(0), vive.getUndistortedTexture(1));

	for (int i = 0; i < 2; i++) {
		
		eyeFbo[i].begin();
		ofClear(0);

		vr.beginEye(vr::Hmd_Eye(i));
		DeepDream.drawLikeViveSRWorks(i, vive.getTransform(i));
		vr.endEye();
		eyeFbo[i].end();

		 //Submit texture to VR!
		vr.submit(eyeFbo[i].getTexture(), vr::EVREye(i));
	}

}


void ofApp::drawVRView() {
	
	ofDisableAlphaBlending();
	eyeFbo[vr::Eye_Left].draw(0, eyeWidth_eighth, eyeWidth_eighth, -eyeWidth_eighth);
	eyeFbo[vr::Eye_Right].draw(eyeWidth_eighth, eyeWidth_eighth, eyeWidth_eighth, -eyeWidth_eighth);
	ofEnableAlphaBlending();
}

void ofApp::draw(){
	
	if (isDrawVrView) {
		drawVRView();
	}
	panel.draw();
	drawInfo();
}

void ofApp::changeDrawVRView() {
	bool b = !isDrawVrView;
	isDrawVrView.set(b);
}

void ofApp::switchDeepDreamThreadProcess() {
	if (DeepDream.getDeepDreamThread().isThreadRunning()) {
		DeepDream.pauseDeepDreamThread();
		ofLogNotice(__FUNCTION__) << "pause deepdream thread.";
	}
	else {
		DeepDream.resumeDeepDreamThread();
		ofLogNotice(__FUNCTION__) << "resume deepdream thread.";
	}
};

void ofApp::keyPressed(int key) {

	if (key == 'd') {
		changeDrawVRView();
		ofLogNotice(__FUNCTION__) << "change Draw VR View";
	}

	if (key == ' ') {
		ofLogNotice(__FUNCTION__) << "spacebar pressed";
		switchDeepDreamThreadProcess();
	}

	//Ctrl + Q
	if (key == 17) {
		ofLogWarning(__FUNCTION__) << "Ctrl+Q : exit.";
		exit();
	}

}

void ofApp::drawInfo() {

	// some information about the timer
	string  info = "FPS:        " + std::to_string(ofGetFrameRate()) + "\n";
	info += "Press 'd' to show/hide VR view.";
	info += "\nPress 'spacebar' to pause/resume DeepDream thread.";
	info += "\nPress 'Ctrl + Q' to exit.";

	ofSetColor(255);
	ofDrawBitmapString(info, 20, 20);

}

void ofApp::exit() {

	vr.exit();
	ofLogVerbose(__FUNCTION__) << "VR Exit.";

	vive.exit();
	ofLogVerbose(__FUNCTION__) << "VIVE Exit.";

	DeepDream.pauseDeepDreamThread();
	DeepDream.exit();

	ofSleepMillis(5000);
	ofLogVerbose(__FUNCTION__) << "ofSleep end and Exit.";

}



