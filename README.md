ofxDeepDream
=====================================
Description
-----------
This is an experimental program for real-time input video conversion with deepdream.
This project is referred to [this github repository](https://github.com/Beinabih/Pytorch-HeadTrip).

Environments
------------
Windows10
Visual Studio 2019 or 2022
CUDA v11.3
openFramewrks v0.11.2
LibTorch v1.11.0
OpenCV v4.5.5 with CUDA

Setup
-----
Pytorch
1. Download LibTorch from https://pytorch.org/get-started/locally/
2. Unzip the downloaded file and copy the contents of the "include" and "lib" directories to the following location.
- libtorch-win-shared-with-deps-1.11.0+cu113/include >> /libs/libtorch/win/include/vs/x64/Release
- libtorch-win-shared-with-deps-1.11.0+cu113/lib >> /libs/libtorch/win/lib/vs/x64/Release
3. Download CUDA from https://developer.nvidia.com/cuda-toolkit-archive/

OpenCV
1. Build openCV with extra CUDA module https://github.com/opencv/opencv_contrib
2. Edit addon_config.mk to change OpenCV paths to your environment.

You don't have to follow the above method, just set up your project to use OpenCV and Libtorch.

Models
1. Download models from or run /pytorch_model_export/model_export.py in Pytorch.
2. Copy Inception_v3.pt and spynet.pt to the /model directory.


