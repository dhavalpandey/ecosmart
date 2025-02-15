import cv2
import numpy as np
import time
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def predict_frame(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    max_index = np.argmax(predictions)
    label = labels[max_index]
    category_name = label.split(' ', 1)[-1] if ' ' in label else label
    confidence = predictions[max_index] * 100
    return category_name, confidence

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

def calibrate_background(cap, roi_top, roi_left, roi_size, duration=30):
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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

roi_top, roi_left, roi_size = 50, 50, 500
baseline_roi = calibrate_background(cap, roi_top, roi_left, roi_size, duration=30)

diff_threshold = 15.0
min_contour_area = 5
cooldown_start_time = None
confirmed_object = False
last_category = None
last_confidence = 0.0

roi_accumulation = []
ACCUM_COUNT = 5
target_shape = None  # will store (width, height) for consistency

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot capture frame")
        break

    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (roi_left, roi_top), (roi_left+roi_size, roi_top+roi_size), (255, 0, 0), 2)

    current_roi = frame[roi_top:roi_top+roi_size, roi_left:roi_left+roi_size]
    current_roi_blurred = cv2.GaussianBlur(current_roi, (5, 5), 0)
    aligned_roi = align_roi(baseline_roi, current_roi_blurred)

    diff = cv2.absdiff(baseline_roi, aligned_roi)
    # Subtract a 5% offset (~13 on a scale of 255)
    diff = cv2.subtract(diff, np.full(diff.shape, 13, dtype=diff.dtype))
    diff = cv2.medianBlur(diff, 5)
    mean_diff = np.mean(diff)

    union_box = None

    if mean_diff > diff_threshold:
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for cnt in contours:
            if cv2.contourArea(cnt) > min_contour_area:
                points.extend(cnt.reshape(-1, 2))
        if points:
            pts = np.array(points)
            x_min = np.percentile(pts[:, 0], 5)
            x_max = np.percentile(pts[:, 0], 95)
            y_min = np.percentile(pts[:, 1], 5)
            y_max = np.percentile(pts[:, 1], 95)
            x, y = int(x_min), int(y_min)
            w, h = int(x_max - x_min), int(y_max - y_min)
            pad = 2
            x = max(x - pad, 0)
            y = max(y - pad, 0)
            if x + w + pad < aligned_roi.shape[1]:
                w = w + pad
            if y + h + pad < aligned_roi.shape[0]:
                h = h + pad
            union_box = (x, y, w, h)
            cv2.rectangle(frame,
                          (roi_left + x, roi_top + y),
                          (roi_left + x + w, roi_top + y + h),
                          (0, 255, 0), 2)

        if not confirmed_object:
            if cooldown_start_time is None:
                cooldown_start_time = time.time()
            elapsed = time.time() - cooldown_start_time
            remaining = max(5 - elapsed, 0)
            box_x = roi_left + union_box[0] if union_box else roi_left + 10
            box_y = roi_top + union_box[1] - 10 if union_box else roi_top + 25
            cv2.putText(frame, f"{remaining:.1f}s", (box_x, box_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if elapsed >= 5 and union_box:
                x, y, w, h = union_box
                obj_roi = aligned_roi[y:y+h, x:x+w]
                # Define the target shape based on the first detection to ensure consistency
                if target_shape is None:
                    target_shape = (w, h)  # (width, height)
                fixed_roi = cv2.resize(obj_roi, target_shape)
                roi_accumulation.append(fixed_roi)
                cv2.putText(frame, f"Accumulating {len(roi_accumulation)}/{ACCUM_COUNT}",
                            (roi_left + x, roi_top + y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if len(roi_accumulation) >= ACCUM_COUNT:
                    avg_roi = np.median(np.array(roi_accumulation), axis=0).astype("uint8")
                    category_name, confidence = predict_frame(avg_roi)
                    last_category, last_confidence = category_name, confidence
                    confirmed_object = True
                    cooldown_start_time = None
                    roi_accumulation = []
                    target_shape = None
                    status_text = f"{category_name} ({confidence:.2f}%)"
                    status_color = (0, 255, 0)
                else:
                    status_text = f"Accumulating {len(roi_accumulation)}/{ACCUM_COUNT}"
                    status_color = (0, 255, 255)
            else:
                status_text = f"Detecting... {remaining:.1f}s"
                status_color = (0, 255, 255)
        else:
            status_text = f"{last_category} ({last_confidence:.2f}%)"
            status_color = (0, 255, 0)
    else:
        cooldown_start_time = None
        confirmed_object = False
        last_category, last_confidence = None, 0.0
        roi_accumulation = []
        target_shape = None
        status_text = "empty"
        status_color = (0, 0, 255)

    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(frame, f"Diff: {mean_diff:.1f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Camera View", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()