#pragma once

#include "ofMain.h"
#include "ofxGui.h"
#include "ofxDeepDreamThread.h"

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
		void keyPressed(int key);

		ofVideoGrabber vidGrabber;
		ofTexture videoTexture;
		ofxPanel panel;
		ofxDeepDream::ofxDeepDreamThread DeepDream;
		
};
