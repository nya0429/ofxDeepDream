#include "ofMain.h"
#include "ofApp.h"

////========================================================================
int main( ){
	ofGLWindowSettings settings;
	settings.setGLVersion(4, 1);
	settings.setSize(960,540);
	//settings.setSize(1920, 1080);

	ofCreateWindow(settings);
	ofRunApp(new ofApp());
    return 0;
}
