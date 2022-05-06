#pragma once

#include "ofMain.h"
#include "ofxOpenVrUtil.h"
#include "ofxViveSRWorks.h"
#include "ofxGui.h"
#include "ofxDeepDreamThread.h"

class ofApp : public ofBaseApp{

public:
	void setup();
	void update();
	void draw();
		
	void exit();
	void keyPressed(int key);

private:

	void allocateTexture();

	int eyeWidth;
	int eyeHeight;
	int eyeWidth_eighth;
	int eyeHeight_eighth;
	void drawVRView();
	void drawInfo();

	//key function
	void changeDrawVRView();
	void switchDeepDreamThreadProcess();

	ofxDeepDream::ofxDeepDreamThread DeepDream;
	ofxOpenVrUtil::Interface vr;
	
	ofxViveSRWorks::Interface vive;
	std::array<ofFbo, 2> eyeFbo;

	ofxPanel panel;
	ofParameter<bool> isDrawVrView;

};
