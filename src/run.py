"""
Entry point for the Robotics Ecosmart Bin Project.
Runs in either development (-dev) or production mode.
"""

import cv2
import numpy as np
import time
import sys
import threading
from utils import init_model, predict_frame, align_roi, calibrate_background
from tray import simulate_tray_rotation

class VideoStream:
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print("Cannot open camera")
            exit()
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# Parameters
if '-dev' in sys.argv:
    CALIBRATION_DURATION = 5
    WAIT_TIME = 3 # seconds object needs to be in frame to be registered
else:
    CALIBRATION_DURATION = 30
    WAIT_TIME = 5
    POST_ROTATION_CALIBRATION = 5

interpreter, input_details, output_details, labels = init_model()
video_stream = VideoStream(src=0, width=640, height=480).start()
time.sleep(1)
roi_top, roi_left, roi_size = 50, 50, 500
baseline_roi = calibrate_background(video_stream.stream, roi_top, roi_left, roi_size, CALIBRATION_DURATION)

diff_threshold = 10.0
min_contour_area = 5
cooldown_start_time = None
confirmed_object = False
last_category = None
last_confidence = 0.0
roi_accumulation = []
ACCUM_COUNT = 5
target_shape = None
detection_enabled = True

prev_frame_time = 0

while True:
    frame = video_stream.read()
    if frame is None:
        continue

    frame = cv2.flip(frame, 1)
    if detection_enabled:
        cv2.rectangle(frame, (roi_left, roi_top), (roi_left + roi_size, roi_top + roi_size), (255, 0, 0), 2)
        current_roi = frame[roi_top:roi_top+roi_size, roi_left:roi_left+roi_size]
        current_roi_blurred = cv2.GaussianBlur(current_roi, (5, 5), 0)
        aligned_roi = align_roi(baseline_roi, current_roi_blurred)
        diff = cv2.absdiff(baseline_roi, aligned_roi)
        diff = cv2.subtract(diff, np.full(diff.shape, 16, dtype=diff.dtype))
        diff = cv2.medianBlur(diff, 5)
        mean_diff = np.mean(diff)

        union_box = None
        if mean_diff > diff_threshold:
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = []
            for cnt in contours:
                if cv2.contourArea(cnt) > min_contour_area:
                    points.extend(cnt.reshape(-1,2))
            if points:
                pts = np.array(points)
                x_min = np.percentile(pts[:, 0], 5)
                x_max = np.percentile(pts[:, 0], 95)
                y_min = np.percentile(pts[:, 1], 5)
                y_max = np.percentile(pts[:, 1], 95)
                x, y = int(x_min), int(y_min)
                w, h = int(x_max - x_min), int(y_max - y_min)
                pad_frac = 0.25
                pad_w = int(w * pad_frac)
                pad_h = int(h * pad_frac)
                x = max(x - pad_w, 0)
                y = max(y - pad_h, 0)
                w = min(w + 2 * pad_w, aligned_roi.shape[1]-x)
                h = min(h + 2 * pad_h, aligned_roi.shape[0]-y)
                union_box = (x, y, w, h)
                cv2.rectangle(frame, (roi_left+x, roi_top+y), (roi_left+x+w, roi_top+y+h), (0,255,0), 2)

            if not confirmed_object:
                if cooldown_start_time is None:
                    cooldown_start_time = time.time()
                elapsed = time.time() - cooldown_start_time
                remaining = max(WAIT_TIME - elapsed, 0)
                box_x = roi_left + union_box[0] if union_box else roi_left+10
                box_y = roi_top + union_box[1]-10 if union_box else roi_top+25
                cv2.putText(frame, f"{remaining:.1f}s", (box_x, box_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                if elapsed >= WAIT_TIME and union_box:
                    x, y, w, h = union_box
                    obj_roi = aligned_roi[y:y+h, x:x+w]
                    if '-dev' in sys.argv:
                        fixed_roi = cv2.resize(obj_roi, (w, h))
                        predicted_category, confidence = predict_frame(fixed_roi, interpreter, input_details, output_details, labels)
                        last_category, last_confidence = predicted_category, confidence
                        confirmed_object = True
                        cooldown_start_time = None
                        status_text = f"{predicted_category} ({confidence:.2f}%)"
                        status_color = (0,255,0)
                    else:
                        if target_shape is None:
                            target_shape = (w, h)
                        fixed_roi = cv2.resize(obj_roi, target_shape)
                        roi_accumulation.append(fixed_roi)
                        cv2.putText(frame, f"Accumulating {len(roi_accumulation)}/{ACCUM_COUNT}", (roi_left+x, roi_top+y-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                        if len(roi_accumulation) >= ACCUM_COUNT:
                            avg_roi = np.median(np.array(roi_accumulation), axis=0).astype("uint8")
                            predicted_category, confidence = predict_frame(avg_roi, interpreter, input_details, output_details, labels)
                            last_category, last_confidence = predicted_category, confidence
                            confirmed_object = True
                            cooldown_start_time = None
                            roi_accumulation = []
                            target_shape = None
                            status_text = f"{predicted_category} ({confidence:.2f}%)"
                            status_color = (0,255,0)
                        else:
                            status_text = f"Accumulating {len(roi_accumulation)}/{ACCUM_COUNT}"
                            status_color = (0,255,255)
                else:
                    status_text = f"Detecting... {remaining:.1f}s"
                    status_color = (0,255,255)
            else:
                status_text = f"{last_category} ({last_confidence:.2f}%)"
                status_color = (0,255,0)
        else:
            cooldown_start_time = None
            confirmed_object = False
            last_category, last_confidence = None, 0.0
            if '-dev' not in sys.argv:
                roi_accumulation = []
                target_shape = None
            status_text = "empty"
            status_color = (0,0,255)
    else:
        status_text = "Detection stopped, bin rotating"
        status_color = (0,0,255)

    cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(frame, f"Diff: {mean_diff:.1f}" if detection_enabled else "", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time) if prev_frame_time else 0
    prev_frame_time = current_time
    frame_width = frame.shape[1]
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Camera View", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    if confirmed_object and detection_enabled:
        detection_enabled = False
        print(f"Object detected: {last_category} ({last_confidence:.2f}%)")
        if last_category in ["plastic", "paper", "food waste", "metal"]:
            simulate_tray_rotation(last_category, frame)
        else:
            print("Invalid category; skipping rotation.")
        if '-dev' not in sys.argv:
            baseline_roi = calibrate_background(video_stream.stream, roi_top, roi_left, roi_size, POST_ROTATION_CALIBRATION)
        confirmed_object = False
        detection_enabled = True

video_stream.stop()
cv2.destroyAllWindows()