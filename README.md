# Robotics Ecosmart Bin Project

This project demonstrates real-time object detection and classification using a TensorFlow Lite model. The system processes camera input by calibrating the background, extracting a region of interest (ROI), and applying advanced image processing and statistical techniques to accurately detect objects.

## Project Structure

Robotics Ecosmart Bin Project/
├── .gitignore
├── README.md // This file
├── requirements.txt
└── src/
├── run.py // Main program for real-time object detection
├── test.py // Test script for evaluating the model on test images
├── model.tflite // TensorFlow Lite model file
├── labels.txt // Labels file with class mappings
├── utils.py // Utility functions for model and image processing
└── tray.py // Functions for tray rotation simulation

## Setup

1. **Python Environment**  
   The project requires Python 3.10. Activate the environment with:
   conda activate py310

2. **Install Dependencies**  
   Install the necessary packages:
   pip install -r requirements.txt

3. **Model and Labels**  
   Place the model.tflite and labels.txt files in the src/ directory.

## How It Works

### 1. Camera Initialization and Background Calibration

-   **Camera Initialization:**  
    The program opens the default camera using OpenCV.

-   **Background Calibration:**  
    The function `calibrate_background()` captures multiple frames over 30 seconds with an empty ROI:
    -   Each frame is blurred to reduce noise.
    -   A median background image is computed and then filtered using a bilateral filter to be used as a reference.

### 2. Frame Processing and ROI Extraction

-   **Frame Acquisition:**

    -   Each frame is captured and flipped horizontally (mirror view).
    -   A fixed rectangular ROI is drawn on the frame.

-   **Preprocessing:**

    -   The current ROI is blurred using a Gaussian filter.
    -   The `align_roi()` function aligns the current ROI with the calibrated background.

-   **Change Detection:**
    -   A difference image is computed between the aligned frame and the background.
    -   A 5% offset (~13 on a 255 scale) is subtracted from the difference image to remove baseline noise.
    -   The image undergoes additional median blurring and thresholding (using Otsu's method) coupled with morphological operations.
    -   If the mean difference exceeds a threshold, the system extracts contours using an advanced statistical method (5th and 95th percentiles) to determine a single tight bounding box around the object.

### 3. Accumulation and Prediction

-   **Frame Accumulation:**

    -   After detecting a stable object (after a 5-second cooldown), the detected region is extracted and resized consistently.
    -   Several frames (as determined by `ACCUM_COUNT`) are accumulated to form a robust median ROI.

-   **Classification:**
    -   Once enough frames are collected, the median ROI is computed and passed into the TensorFlow Lite model via `predict_frame()`.
    -   The predicted object category and its confidence are displayed on the camera view.

### 4. Tray Rotation Simulation

-   **Tray Rotation:**
    -   When an object is confirmed, the `simulate_tray_rotation()` function is called.
    -   The function calculates the shortest path between compartments using a graph and simulates the tray rotation.
    -   During rotation, detection is disabled, and messages are displayed on the camera view.

### 5. User Interaction

-   **Real-Time Feedback:**  
    The live camera view shows:

    -   The ROI and bounding box.
    -   Status messages like "Detecting...", "Accumulating...", or the final predicted category with confidence.
    -   The computed mean difference value.

-   **Exiting the Program:**  
    Press 'q' to gracefully exit the application. The camera is released and OpenCV windows are closed.

## Testing

A separate script (test.py) is provided under src/tests/:

-   It processes a collection of test images, applies similar preprocessing, and then uses the model to predict object categories.
-   Test performance (such as accuracy) is computed from the results.

## Training Notes

-   The model was trained for 30 epochs on a dataset consisting of 14,617 images.
-   Note: Although Label ID 2 was initially "glass", the system now treats glass and plastic as the same category.

Enjoy exploring the Robotics Ecosmart Bin Project!
