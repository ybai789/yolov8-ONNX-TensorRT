
#include <opencv2/opencv.hpp>

#include "yolo.hpp"

#include <iostream>
#include <filesystem>
#include "chrono"

#include <sys/stat.h>
#include <cstring>

using namespace std;

static const char *cocolabels[] = {"person",        "bicycle",      "car",
                                   "motorcycle",    "airplane",     "bus",
                                   "train",         "truck",        "boat",
                                   "traffic light", "fire hydrant", "stop sign",
                                   "parking meter", "bench",        "bird",
                                   "cat",           "dog",          "horse",
                                   "sheep",         "cow",          "elephant",
                                   "bear",          "zebra",        "giraffe",
                                   "backpack",      "umbrella",     "handbag",
                                   "tie",           "suitcase",     "frisbee",
                                   "skis",          "snowboard",    "sports ball",
                                   "kite",          "baseball bat", "baseball glove",
                                   "skateboard",    "surfboard",    "tennis racket",
                                   "bottle",        "wine glass",   "cup",
                                   "fork",          "knife",        "spoon",
                                   "bowl",          "banana",       "apple",
                                   "sandwich",      "orange",       "broccoli",
                                   "carrot",        "hot dog",      "pizza",
                                   "donut",         "cake",         "chair",
                                   "couch",         "potted plant", "bed",
                                   "dining table",  "toilet",       "tv",
                                   "laptop",        "mouse",        "remote",
                                   "keyboard",      "cell phone",   "microwave",
                                   "oven",          "toaster",      "sink",
                                   "refrigerator",  "book",         "clock",
                                   "vase",          "scissors",     "teddy bear",
                                   "hair drier",    "toothbrush"};

const std::vector<std::string> CLASS_NAMES = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus",
	"train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat",
	"dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella",
	"handbag", "tie", "suitcase", "frisbee", "skis",
	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
	"cup", "fork", "knife", "spoon", "bowl",
	"banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table",
	"toilet", "tv", "laptop", "mouse", "remote",
	"keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush" };

const std::vector<std::vector<unsigned int>> COLORS = {
	{ 0, 114, 189 }, { 217, 83, 25 }, { 237, 177, 32 },
	{ 126, 47, 142 }, { 119, 172, 48 }, { 77, 190, 238 },
	{ 162, 20, 47 }, { 76, 76, 76 }, { 153, 153, 153 },
	{ 255, 0, 0 }, { 255, 128, 0 }, { 191, 191, 0 },
	{ 0, 255, 0 }, { 0, 0, 255 }, { 170, 0, 255 },
	{ 85, 85, 0 }, { 85, 170, 0 }, { 85, 255, 0 },
	{ 170, 85, 0 }, { 170, 170, 0 }, { 170, 255, 0 },
	{ 255, 85, 0 }, { 255, 170, 0 }, { 255, 255, 0 },
	{ 0, 85, 128 }, { 0, 170, 128 }, { 0, 255, 128 },
	{ 85, 0, 128 }, { 85, 85, 128 }, { 85, 170, 128 },
	{ 85, 255, 128 }, { 170, 0, 128 }, { 170, 85, 128 },
	{ 170, 170, 128 }, { 170, 255, 128 }, { 255, 0, 128 },
	{ 255, 85, 128 }, { 255, 170, 128 }, { 255, 255, 128 },
	{ 0, 85, 255 }, { 0, 170, 255 }, { 0, 255, 255 },
	{ 85, 0, 255 }, { 85, 85, 255 }, { 85, 170, 255 },
	{ 85, 255, 255 }, { 170, 0, 255 }, { 170, 85, 255 },
	{ 170, 170, 255 }, { 170, 255, 255 }, { 255, 0, 255 },
	{ 255, 85, 255 }, { 255, 170, 255 }, { 85, 0, 0 },
	{ 128, 0, 0 }, { 170, 0, 0 }, { 212, 0, 0 },
	{ 255, 0, 0 }, { 0, 43, 0 }, { 0, 85, 0 },
	{ 0, 128, 0 }, { 0, 170, 0 }, { 0, 212, 0 },
	{ 0, 255, 0 }, { 0, 0, 43 }, { 0, 0, 85 },
	{ 0, 0, 128 }, { 0, 0, 170 }, { 0, 0, 212 },
	{ 0, 0, 255 }, { 0, 0, 0 }, { 36, 36, 36 },
	{ 73, 73, 73 }, { 109, 109, 109 }, { 146, 146, 146 },
	{ 182, 182, 182 }, { 219, 219, 219 }, { 0, 114, 189 },
	{ 80, 183, 189 }, { 128, 128, 0 }
};

yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }


inline bool IsFile(const std::string& path)
{
    struct stat path_stat;
    if (stat(path.c_str(), &path_stat) != 0)
    {
        std::cerr << __FILE__ << ":" << __LINE__ << " " << path << " does not exist or error occurred" << std::endl;
        return false;
    }
    if (S_ISDIR(path_stat.st_mode))
    {
        std::cerr << __FILE__ << ":" << __LINE__ << " " << path << " is a directory" << std::endl;
        return false;
    }
    return true;
}

inline bool IsFolder(const std::string& path)
{
    struct stat path_stat;
    if (stat(path.c_str(), &path_stat) != 0)
    {
        return false; // Error occurred or the file/directory does not exist
    }
    return S_ISDIR(path_stat.st_mode);
}

void draw_objects(const cv::Mat& image,
                  cv::Mat& res,
                  yolo::BoxArray& objs,
                  const std::vector<std::string>& CLASS_NAMES,
                  const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();

    for (auto& obj : objs)
    {
        int idx = obj.class_label;
        cv::Scalar color = cv::Scalar(
            COLORS[idx][0],
            COLORS[idx][1],
            COLORS[idx][2]
        );

        cv::Rect rect(static_cast<int>(obj.left), static_cast<int>(obj.top),
                      static_cast<int>(obj.right - obj.left), static_cast<int>(obj.bottom - obj.top));

        cv::rectangle(
            res,
            rect,
            color,
            2
        );

        char text[256];
        sprintf(
            text,
            "%s %.1f%%",
            CLASS_NAMES[idx].c_str(),
            obj.confidence * 100
        );

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(
            text,
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            1,
            &baseLine
        );

        int x = rect.x;
        int y = rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(
            res,
            cv::Rect(x, y, label_size.width, label_size.height + baseLine),
            color, 
            -1
        );

        cv::putText(
            res,
            text,
            cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            { 255, 255, 255 },
            1
        );
    }
}



int main(int argc, char** argv)
{

  std::cout << "running main..." << std::endl;
  const std::string engine_file_path{ argv[1] };
  const std::string path{ argv[2] };

  std::vector<std::string> imagePathList;
  bool isVideo{ false };

  assert(argc == 3);

  auto yolo = yolo::load(engine_file_path, yolo::Type::V8);
  if (yolo == nullptr) {
      std::cout << "Failed to load yolo model." << std::endl;
      return 1;
  }

	if (IsFile(path))
	{
		std::string suffix = path.substr(path.find_last_of('.') + 1);
		if (
			suffix == "jpg" ||
				suffix == "jpeg" ||
				suffix == "png"
			)
		{
			imagePathList.push_back(path);
		}
		else if (
			suffix == "mp4" ||
				suffix == "avi" ||
				suffix == "m4v" ||
				suffix == "mpeg" ||
				suffix == "mov" ||
				suffix == "mkv"
			)
		{
			isVideo = true;
		}
		else
		{
			printf("suffix %s is wrong !!!\n", suffix.c_str());
			std::abort();
		}
	}
	else if (IsFolder(path))
	{
		cv::glob(path + "/*.jpg", imagePathList);
	}

  cv::Mat res, image;
  //cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

  if (isVideo)
  {
		cv::VideoCapture cap(path);

		if (!cap.isOpened())
		{
			printf("can not open %s\n", path.c_str());
			return -1;
		}

    // Get the frame size from the video input
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Size frameSize(frameWidth, frameHeight);

    std::string outputDirectory = "output/";
    std::size_t fileNameStart = path.find_last_of("/\\") + 1;
    std::size_t dotPos = path.find_last_of(".");
    std::string outputVideoPath = outputDirectory + path.substr(fileNameStart, dotPos - fileNameStart) + "_det.mp4";

    // Example for MP4V codec
    int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    // Get the fps from the video input
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Create the VideoWriter object
    cv::VideoWriter videoWriter(outputVideoPath, codec, fps, frameSize);

	while (cap.read(image))
	 {
        auto start = std::chrono::system_clock::now();
        auto objs = yolo->forward(cvimg(image));
        auto end = std::chrono::system_clock::now();
        auto tc = (double)
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        printf("inference %2.4lf ms\n", tc);
        draw_objects(image, res, objs, CLASS_NAMES, COLORS);

        videoWriter.write(res);
		}
	}
	else // not video
	{
		for (auto& path : imagePathList)
		{

      image = cv::imread(path);

      auto start = std::chrono::system_clock::now();
      auto objs = yolo->forward(cvimg(image));
      auto end = std::chrono::system_clock::now();
      auto tc = (double)
				std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
			printf("inference %2.4lf ms\n", tc);
      draw_objects(image, res, objs, CLASS_NAMES, COLORS);

      std::string outputDirectory = "output/";
      std::size_t fileNameStart = path.find_last_of("/\\") + 1;
      std::size_t dotPos = path.find_last_of(".");
      std::string outputImagePath = outputDirectory + path.substr(fileNameStart, dotPos - fileNameStart) + "_det" + path.substr(dotPos);

      cv::imwrite(outputImagePath, res);

		}
	}
	cv::destroyAllWindows();
	return 0;
}



extern "C" void* Init(const char* model_path) {
    std::string engine_name = model_path;
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.5f;
    auto yolo = loadraw(engine_name, yolo::Type::V8, confidence_threshold, nms_threshold);
    return yolo;
}

extern "C" void Detect(void* p, int rows, int cols, unsigned char* src_data, float(*res_array)[6]) {
    cv::Mat frame = cv::Mat(rows, cols, CV_8UC3, src_data);
    yolo::Infer* yolov8 = (yolo::Infer*)p;
    auto objs = yolov8->forward(cvimg(frame));
    int i = 0;
    for (auto &obj : objs) {
        res_array[i][0] = obj.left;
        res_array[i][1] = obj.top;
        res_array[i][2] = obj.right - obj.left;
        res_array[i][3] = obj.bottom - obj.top;;
        res_array[i][4] = obj.class_label;
        res_array[i][5] = obj.confidence;
        i++;
    }
}


