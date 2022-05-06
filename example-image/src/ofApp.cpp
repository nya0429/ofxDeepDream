#include "ofApp.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

//--------------------------------------------------------------
void ofApp::setup() {

    ofSetLogLevel(OF_LOG_VERBOSE);
    ofLogToConsole();

    ofDisableArbTex();
    ofSetVerticalSync(false);
    ofSetFrameRate(0);
    panel.setup();
    panel.setPosition(800,0);
    panel.add(DeepDream.getParameters());

}

//--------------------------------------------------------------
void ofApp::update() {
}

//--------------------------------------------------------------
void ofApp::draw() {

    ofDrawBitmapString("Press \"o\" to open an image, spacebar to run deepdream, \"r\" to reset an image.", 20, 15);

    if (DeepDream.isSetupTexture()) {
        ofDisableAlphaBlending();
        DeepDream.getTexture().draw(20, 20);
        ofEnableAlphaBlending();
    }
    panel.draw();
    
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {

    if (key == ' ') {
        DeepDream.single_process();
    }

    if (key == 'o') {
        //Open the Open File Dialog
        ofFileDialogResult openFileResult = ofSystemLoadDialog("Select a jpg or png");
        //Check if the user opened a file
        if (openFileResult.bSuccess) {
            ofLogVerbose("User selected a file");
            //We have a file, check it and process it
            processOpenFileSelection(openFileResult);
        }
        else {
            ofLogVerbose("User hit cancel");
        }
    }

    if (key == 'r') {
        DeepDream.reset();
    }

}


void ofApp::processOpenFileSelection(ofFileDialogResult openFileResult) {

	ofLogVerbose("getName(): " + openFileResult.getName());
	ofLogVerbose("getPath(): " + openFileResult.getPath());

	ofFile file(openFileResult.getPath());

	if (file.exists()) {

		ofLogVerbose("The file exists - now checking the type via file extension");
		string fileExtension = ofToUpper(file.getExtension());

		//We only want images
		if (fileExtension == "JPG" || fileExtension == "PNG") {

			//Load the selected image
			ofImage image;
			image.load(openFileResult.getPath());
            image.setImageType(OF_IMAGE_COLOR_ALPHA);
            imageTexture.loadData(image);
            DeepDream.setup(imageTexture);
            DeepDream.pauseDeepDreamThread();
            DeepDream.stop();

		}
	}

}
//--------------------------------------------------------------