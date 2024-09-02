

This repository offers a method to accelerate YOLOv8 object detection for GPU execution, covering the entire pipeline: preprocessing (image warping), inference, and postprocessing (NMS). This is achieved by converting the model weights to ONNX format and then compiling them into a TensorRT engine.

## 1 Prepare the environment

1. Install `CUDA` follow [`CUDA official website`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-the-nvidia-cuda-toolkit).

   🚀 RECOMMENDED `CUDA` >= 11.8

2. Install `TensorRT` follow [`TensorRT official website`](https://developer.nvidia.com/nvidia-tensorrt-8x-download).

   🚀 RECOMMENDED `TensorRT` >= 8.6


3. Install YOLOv8 follow  [`YOLOv8 official website`](https://github.com/ultralytics/ultralytics).  

4. Prepare your own PyTorch weight such as `yolov8s.pt`.

## 2 Convert YOLOv8 weight to ONNX model  and TensorRT engine


1. Export ONNX Weight File

```bash
yolo export model= yolov8s.pt format=onnx simplify opset=11
```

Copy the generated `yolov8s.onnx` file to the `workspace` directory.

2. Convert to ONNX Model Using `transform.py`

Run the following command in the `workspace` directory:

```bash
python transform.py yolov8s.onnx
```

This will generate the `yolov8s.transd.onnx` file.

3. Convert ONNX Model to TensorRT Engine File using **trtexec**

Remove the `--fp16` parameter if you prefer to use FP32 for inference.

```bash
trtexec --onnx=./yolov8s.transd.onnx --saveEngine=./yolov8s.engine --fp16
```

## 3 Compiling C++ program

1. Modify `CMakeLists.txt`

Edit the `CMakeLists.txt` file to adjust the configurations according to the software locations and GPU architecture on your computer.

To find your GPU architecture, refer to the official website: [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus).

2. Modify `src/detect.cpp` according to your customized dataset

3. Compile

```bash
mkdir build
cd build
cmake ..
make
```

## 4 Execute YOLOv8 with TensorRT acceleration (C++)

Copy `detect` and `libyolov8.so` from the `build` directory to the `workspace` directory.

**Image Inference**

```bash
./detect yolov8s.engine images/your_image_file.jpg
```

**Video  Inference**

```bash
./detect yolov8s.engine video/your_video_file.mp4
```

## 5 Execute YOLOv8 with TensorRT acceleration (Python)

Copy `libyolov8.so` from the `build` directory to the `workspace` directory.

**Image Inference**

```bash
python detect_trt.py --engine ./yolov8s.engine --imgs ./images/your_image_file.jpg --out-dir output
```

**Video Inference**

```bash
python detect_trt.py --engine ./yolov8s.engine --imgs ./video/your_video_file.mp4  --out-dir output
```