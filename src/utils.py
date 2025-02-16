"""
Contains all code related to model initialization, prediction, ROI alignment, and background calibration.
"""

import cv2
import numpy as np
import tensorflow as tf
import time

# default using quantized model, change file name for unquantized model
def init_model(model_path="model_quant.tflite", labels_path="labels.txt"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return interpreter, input_details, output_details, labels

def predict_frame(image, interpreter, input_details, output_details, labels):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    input_dtype = input_details[0]['dtype']
    # Process image based on model's expected input type.
    if input_dtype == np.uint8:
        img = np.array(img, dtype=np.uint8)
    else:
        img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Dequantize predictions if necessary.
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        predictions = (predictions - zero_point) * scale

    max_index = np.argmax(predictions)
    label = labels[max_index]
    category = label.split(' ', 1)[-1] if ' ' in label else label
    # If input was quantized, the predictions have been scaled already.
    confidence = predictions[max_index] * 100 if input_dtype != np.uint8 else predictions[max_index] * 100
    return category.lower(), confidence

def align_roi(baseline, current):
    baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        cv2.findTransformECC(baseline_gray, current_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        aligned = cv2.warpAffine(current, warp_matrix, (current.shape[1], current.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned
    except cv2.error:
        return current

def calibrate_background(cap, roi_top, roi_left, roi_size, duration):
    print("Calibrating. Ensure ROI is empty...")
    frames = []
    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        roi = frame[roi_top:roi_top+roi_size, roi_left:roi_left+roi_size]
        roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        frames.append(roi_blurred.astype("float"))
        cv2.imshow("Calibration ROI", roi)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
    cv2.destroyWindow("Calibration ROI")
    median_frame = np.median(frames, axis=0).astype("uint8")
    return cv2.bilateralFilter(median_frame, d=9, sigmaColor=75, sigmaSpace=75)