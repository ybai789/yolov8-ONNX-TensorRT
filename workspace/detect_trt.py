from ctypes import *
import cv2
import numpy as np
import numpy.ctypeslib as npct
import platform
import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Union
import random


random.seed(0)

# detection model classes
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# colors for per classes
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}


# image suffixs
SUFFIXS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff',
           '.gif', '.webp', '.pfm')
           
VIDEO_SUFFIXS = ('.mp4', '.mkv', '.flv', '.avi', '.mov', '.wmv')
           
def path_to_list(images_path: Union[str, Path]) -> List:
    if isinstance(images_path, str):
        images_path = Path(images_path)
    assert images_path.exists()
    if images_path.is_dir():
        images = [
            i.absolute() for i in images_path.iterdir() if i.suffix in SUFFIXS
        ]
    else:
        assert images_path.suffix in SUFFIXS
        images = [images_path.absolute()]
    return images
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args
    
class Detector():
    def __init__(self, model_path, dll_path):
        if platform.system().lower() == 'windows':
            self.yolov8 = CDLL(dll_path, winmode=0)
        else:
            self.yolov8 = cdll.LoadLibrary(dll_path)
        self.yolov8.Detect.argtypes = [c_void_p, c_int, c_int, POINTER(c_ubyte), npct.ndpointer(dtype=np.float32, ndim=2, shape=(100, 6), flags="C_CONTIGUOUS")]
        self.yolov8.Init.restype = c_void_p
        self.yolov8.Init.argtypes = [c_void_p]
        self.c_point = self.yolov8.Init(model_path)

    def predict(self, img):
        rows, cols = img.shape[0], img.shape[1]
        res_arr = np.zeros((100, 6), dtype=np.float32)
        self.yolov8.Detect(self.c_point, c_int(rows), c_int(cols), img.ctypes.data_as(POINTER(c_ubyte)), res_arr)
        self.bbox_array = res_arr[~(res_arr == 0).all(1)]
        return self.bbox_array

    
def visualize(img, bbox_array):
    for bbox_data in bbox_array:
        x, y, w, h = bbox_data[0:4]  # 解包 xywh
        cls_id = int(bbox_data[4])
        score = bbox_data[5]
        cls = CLASSES[cls_id]        # 假设 CLASSES 是类名的列表
        color = COLORS[cls]       # 假设 COLORS 是与类名对应的颜色列表

        # 绘制边框和文本
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        label = f"{cls} {score:.2f}"
        img = cv2.putText(img, label, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return img

def infer_video(input_file_path, det, save_path):

    # Open the video file
    cap = cv2.VideoCapture(input_file_path)

    if not cap.isOpened():
        print(f"Cannot open {input_file_path}")
        exit(-1)

    # Get the frame size from the video input
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameSize = (frameWidth, frameHeight)


    # Assuming input_file_path is a string with the file path
    input_file_path_obj = Path(input_file_path)

    # Extract the file stem (name without extension)
    file_stem = input_file_path_obj.stem

    # Now you can create the output video path
    outputVideoPath = str(save_path / file_stem) + "_det.mp4"


    # Define the output video file
    #dotPos = input_file_path.rfind(".")
    #outputVideoPath = str(save_path / input_file_path[:dotPos]) + "_det.mp4"

    # Example for MP4V codec
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # Get the fps from the video input
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create the VideoWriter object
    videoWriter = cv2.VideoWriter(outputVideoPath, codec, fps, frameSize)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        
        result = det.predict(frame)
        
        img = visualize(frame, result)
            
        end = time.time()
        frameprocess_time = (end - start) * 1000  
        print(f"frameprocess_time: {frameprocess_time:.2f} ms")

        videoWriter.write(img)

    # Release resources
    cap.release()
    videoWriter.release()
    print('input->{} saved'.format(outputVideoPath))
    
def main(args: argparse.Namespace) -> None:
    model_path = args.engine.encode()  # Convert model_path to byte string
    input_file_path = args.imgs

    print("Model Path:", model_path)
    print("Images Path:", input_file_path)

    if platform.system().lower() == 'windows':
        DLL_PATH =  "./yolov8.dll"
    else:
        DLL_PATH =  "./libyolov8.so"

    # 创建 Detector 实例，传递 model_path 作为字节字符串
    det = Detector(model_path=model_path, dll_path=DLL_PATH)
    
    isFolder = False
    isImage = False
    isVideo = False
    if os.path.isdir(input_file_path):
        isFolder = True  # 是一个目录
    elif os.path.isfile(input_file_path):
        file_extension = os.path.splitext(input_file_path.lower())[1]
        if file_extension in SUFFIXS:
            isImage = True  # 是一个图像文件
        elif file_extension in VIDEO_SUFFIXS:
            isVideo = True  # 是一个视频文件
        else:
            print("It's a file, but not an image or video file.")
    else:
        print("The path is neither a directory nor a file.")

    save_path = Path(args.out_dir)
    
    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    if isImage or isFolder:
        images = path_to_list(input_file_path) 

    if isVideo:
        infer_video(input_file_path, det, save_path)
        exit()

    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Unable to load image at {image_path}")
            continue

        start = time.time()
        result = det.predict(img)
        end = time.time()
        inference_time = (end - start) * 1000  
        print(f"Inference time for {image_path}: {inference_time:.2f} ms")
        
        img = visualize(img, result)
        if args.show:
            cv2.imshow(str(image_path.name), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        save_image_path = save_path / image_path.name
        cv2.imwrite(str(save_image_path), img)
        print(f"Saved image at: {save_image_path}")  # 打印保存图像的路径

if __name__ == "__main__":
    args = parse_args()
    main(args)


