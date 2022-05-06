meta:
    ADDON_NAME = ofxDeepDream
    ADDON_AUTHOR = @nya0429
    ADDON_URL = https://github.com/

common:

    ADDON_INCLUDES =
    ADDON_INCLUDES += "src"
    ADDON_DEFINES = "AT_PARALLEL_OPENMP=1"
    ADDON_LIBS =

vs:
    #openCV
    ADDON_INCLUDES += "C:/lib/OpenCV455/include"
    ADDON_LIBS += "C:/lib/OpenCV455/x64/vc16/lib/opencv_world455.lib"
	ADDON_DLLS_TO_COPY += "C:/lib/OpenCV455/x64/vc16/bin/opencv_world455.dll"

    # libtorch
    ADDON_INCLUDES += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include"

    # Release
    ADDON_INCLUDES += "libs/win/include/vs/x64/Release"
    ADDON_INCLUDES += "libs/win/include/vs/x64/Release/torch/csrc/api/include"
    # Debug
    # ADDON_INCLUDES += "libs/win/include/vs/x64/Debug"
    # ADDON_INCLUDES += "libs/win/include/vs/x64/Debug/torch/csrc/api/include"

    ADDON_LIBS += "libs/win/lib/vs/x64/Release/caffe2_nvrtc.lib"
    ADDON_LIBS += "c10.lib"
    ADDON_LIBS += "c10_cuda.lib"
    ADDON_LIBS += "torch.lib"
    ADDON_LIBS += "torch_cpu.lib"
    ADDON_LIBS += "torch_cuda.lib"
    ADDON_LIBS += "torch_cuda_cpp.lib"
    ADDON_LIBS += "torch_cuda_cu.lib"
    ADDON_LIBS += "-INCLUDE:?warp_size@cuda@at@@YAHXZ"
    ADDON_LIBS += "-INCLUDE:?_torch_cuda_cu_linker_symbol_op_cuda@native@at@@YA?AVTensor@2@AEBV32@@Z"
    ADDON_LIBS += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cudart.lib"
    ADDON_LIBS += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cufft.lib"
    ADDON_LIBS += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/curand.lib"
    ADDON_LIBS += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cublas.lib"
    ADDON_LIBS += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64/cudnn.lib"

