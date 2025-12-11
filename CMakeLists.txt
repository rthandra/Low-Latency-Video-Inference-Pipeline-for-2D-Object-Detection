cmake_minimum_required(VERSION 3.10)
project(YOLOv11_OBB)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Apple Silicon Support
set(CMAKE_OSX_ARCHITECTURES "arm64")

# ------------------ OPEN CV ------------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ------------------ ONNX RUNTIME (MANUAL PATHS) ------------------
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime-osx-arm64-1.23.2")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")

message(STATUS "Using ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Using ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# ------------------ INCLUDE PROJECT HEADERS ------------------
include_directories("include/")

# ------------------ EXECUTABLES ------------------
add_executable(image_inference src/image_inference.cpp)
add_executable(video_inference src/video_inference.cpp)
add_executable(camera_inference src/camera_inference.cpp)

# ------------------ LINK LIBRARIES ------------------
foreach(target image_inference video_inference camera_inference)
    target_link_libraries(${target} ${OpenCV_LIBS})
    target_link_libraries(${target} ${ONNXRUNTIME_LIBRARY})
endforeach()
