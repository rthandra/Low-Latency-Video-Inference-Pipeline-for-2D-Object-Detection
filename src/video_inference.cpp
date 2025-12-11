/**
 * @file video_inference.cpp
 * @brief Object detection in a video stream using YOLOv8 and YOLOv11 in OBB format.
 * 
 * This file implements an object detection application that utilizes YOLO 
 * (You Only Look Once) models, specifically versions 8 and 11, for the 
 * Oriented Bounding Box (OBB) format. 
 * The application processes a video stream to detect objects and saves 
 * the results to a new video file with oriented bounding boxes around detected objects.
 *
 * The application supports the following functionality:
 * - Loading a video stream from disk or camera.
 * - Initializing the YOLOv8 or YOLOv11 OBB detector with the desired model and labels.
 * - Detecting objects within each frame of the video.
 * - Drawing oriented bounding boxes around detected objects and saving the result.
 *
 * Configuration parameters can be adjusted to suit specific requirements:
 * - `isGPU`: Set to true to enable GPU processing for improved performance; 
 *   set to false for CPU processing.
 * - `labelsPath`: Path to the class labels file (e.g., Dota dataset).
 * - `videoPath`: Path to the input video file (e.g., input.mp4).
 * - `outputPath`: Path for saving the output video file (e.g., output.mp4).
 * - `modelPath`: Path to the YOLO model file (e.g., ONNX format).
 *
 * Usage Instructions:
 * 1. Compile the application with the necessary OpenCV and YOLO dependencies.
 * 2. Ensure that the specified video and model files are present in the 
 *    provided paths.
 * 3. Run the executable to initiate the object detection process.
 *
 * Author: Megha Agrawal, Rushika Thandra
 * Date: 11.18.2025
 *
 */

 
// Include necessary headers
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "../include/YOLO11-OBB.hpp" 
#include <chrono>
// Thread-safe queue implementation
template <typename T>
class SafeQueue {
public:
    SafeQueue() : q(), m(), c() {}

    // Add an element to the queue.
    void enqueue(T t) {
        std::lock_guard<std::mutex> lock(m);
        q.push(t);
        c.notify_one();
    }

    // Get the first element from the queue.
    bool dequeue(T& t) {
        std::unique_lock<std::mutex> lock(m);
        while (q.empty()) {
            if (finished) return false;
            c.wait(lock);
        }
        t = q.front();
        q.pop();
        return true;
    }

    void setFinished() {
        std::lock_guard<std::mutex> lock(m);
        finished = true;
        c.notify_all();
    }

private:
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable c;
    bool finished = false;
};

int main(){

    // Timing variables
    double totalComputeTime = 0.0;
    int computeFrames = 0;

    // Paths to the model, labels, input video, and output video
    const std::string labelsPath = "../../bdd100k_classes.txt";
    const std::string videoPath = "../../input_video.mov"; // Input video path
    const std::string outputPath = "../../output_video.mov"; // Output video path
    
    const std::string modelPath = "/Users/rthandra/Desktop/Yolov11_bdd100k_onnx_inference/yolo_finetuned_bdd100k_lightweight.onnx"; // V12
    //const std::string modelPath = "../../yolo_finetuned_bdd100k.onnx"; //V11
    // const std::string modelPath = "../models/yolo11n-obb.onnx"; //V11
    // const std::string modelPath = "../models/yolov8n-obb.onnx"; //V8

    // Initialize the YOLO detector
    bool isGPU = true; // Set to false for CPU processing
    YOLO11OBBDetector detector(modelPath, labelsPath, isGPU); 


    // Open the video file
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()){
        std::cerr << "Error: Could not open or find the video file!\n";
        return -1;
    }

    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC)); // Get codec of input video

    // Target processing/output size
    //int outWidth = 320;
    //int outHeight = 320;

// Create a VideoWriter object to save the output video at 640x640
    cv::VideoWriter out(
        outputPath,
        cv::VideoWriter::fourcc('m','p','4','v'),
        fps,
        //cv::Size(outWidth, outHeight),
        cv::Size(frameWidth, frameHeight),
        true
    );
    if (!out.isOpened()){
        std::cerr << "Error: Could not open the output video file for writing!\n";
        return -1;
    }


    // Thread-safe queues and processing...
    // Thread-safe queues
    SafeQueue<cv::Mat> frameQueue;
    SafeQueue<std::pair<int, cv::Mat>> processedQueue;

    // Flag to indicate processing completion
    std::atomic<bool> processingDone(false);
   // std::cout<< "processing done" << std::endl;

    // Capture thread
    std::thread captureThread([&]() {
        cv::Mat frame;
        int frameCount = 0;
        while (cap.read(frame)){
            std::cout << "[CAPTURE] frame " << frameCount << std::endl;
            frameQueue.enqueue(frame.clone()); // Clone to ensure thread safety
            frameCount++;
        }
        frameQueue.setFinished();
    });

    std::cout<< "captureThread is launched" << std::endl;
    // Processing thread
    std::thread processingThread([&]() {
        cv::Mat frame;
        int frameIndex = 0;
        //int skipFactor = 3; // Process every 3rd frame
        
        while (frameQueue.dequeue(frame)){
             //Skip frames to speed up inference
            /*if (frameIndex % skipFactor != 0) {
                frameIndex++;
                continue;
            }*/
            std::cout << "[PROCESS] frame " << frameIndex << std::endl;

            //Resize frame to 640x640 before detection
            //cv::resize(frame, frame, cv::Size(outWidth, outHeight));
         
            // Detect objects in the frame
            //std::vector<Detection> results = detector.detect(frame);
                // --- YOLO Compute Time Measurement Start ---
            auto computeStart = std::chrono::high_resolution_clock::now();

            // Detect objects in the frame (YOLO inference)
            std::vector<Detection> results = detector.detect(frame);

            auto computeEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> computeDiff = computeEnd - computeStart;

            totalComputeTime += computeDiff.count();
            computeFrames++;
// --- YOLO Compute Time Measurement End ---

            // Draw bounding boxes on the frame
            detector.drawBoundingBox(frame, results); // Uncomment for mask drawing

            // Enqueue the processed frame
            processedQueue.enqueue(std::make_pair(frameIndex++, frame));
        }
        processedQueue.setFinished();
    });

    std::cout<< "processing Thread is launched" << std::endl;
    // Writing thread
    std::thread writingThread([&]() {
        std::pair<int, cv::Mat> processedFrame;
        while (processedQueue.dequeue(processedFrame)){
            std::cout << "[WRITE] frame " << processedFrame.first << std::endl;
            out.write(processedFrame.second);
        }
    });

    std::cout<< "writing Thread is launched" << std::endl;
    // Wait for all threads to finish
    captureThread.join();
    processingThread.join();
    writingThread.join();

    // Print timing results
    std::cout << "\n          COMPUTE TIME REPORT         \n";
    std::cout << "Total YOLO Compute Time (inference only): " 
          << totalComputeTime << " seconds\n";

    if (computeFrames > 0) {
        std::cout << "Frames processed (YOLO inference): " 
              << computeFrames << "\n";

        std::cout << "Average time per frame: "
              << (totalComputeTime / computeFrames) << " seconds\n";
}

    // Release resources
    cap.release();
    out.release();
    cv::destroyAllWindows();

    std::cout << "Video processing completed successfully." << std::endl;

    return 0;
}
