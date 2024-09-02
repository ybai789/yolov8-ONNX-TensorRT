cmake_minimum_required(VERSION 3.18)

project(yolov8 LANGUAGES CXX CUDA) 

#change to your own path
##################################################
set(OpenCV_DIR "/home/bai/opencv-4.7.0/build")  
set(TRT_DIR "/home/bai/TensorRT-8.6.1.6")  
##################################################

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 86)  # Replace 86 with your target architecture(s)
###########################################################################################
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-g;-G;-gencode)
###########################################################################################

####
enable_language(CUDA)  # add this line, then no need to setup cuda path in vs
####
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TRT_DIR}/include)
include_directories(${TRT_DIR}/samples/common)

# -D_MWAITXINTRIN_H_INCLUDED for solving error: identifier "__builtin_ia32_mwaitx" is undefined
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

# setup opencv
find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${TRT_DIR}/lib)

file(GLOB CUDACPPS 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
  )


add_library(${PROJECT_NAME} SHARED ${CUDACPPS})
target_link_libraries(${PROJECT_NAME} "nvinfer" "nvinfer_plugin" "nvonnxparser" "nvparsers")  
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})     
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})  
target_link_libraries(${PROJECT_NAME} Threads::Threads)   


add_executable(detect ${CUDACPPS})

target_link_libraries(detect "nvinfer" "nvinfer_plugin" "nvonnxparser" "nvparsers")  
target_link_libraries(detect ${OpenCV_LIBS})     
target_link_libraries(detect ${CUDA_LIBRARIES})  
target_link_libraries(detect Threads::Threads) 
target_link_libraries(detect ${GLIB2_LIBRARIES})

