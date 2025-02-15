import numpy as np
import tensorflow as tf
import cv2
import os

interpreter = tf.lite.Interpreter(model_path="../model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("../labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0) 
    return img

def get_actual_category(file_path):
    return os.path.basename(os.path.dirname(file_path))

def predict(image_path):
    img = preprocess_image(image_path)
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    max_index = np.argmax(predictions)
    label = labels[max_index]
    
    category_name = label.split(' ', 1)[-1] if ' ' in label else label
    confidence = predictions[max_index]
    return category_name, confidence*100

test_data_dir = '.'
total_files = 0
correct_predictions = 0

for root, dirs, files in os.walk(test_data_dir):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file)
            actual_category = get_actual_category(file_path)
            predicted_category, confidence = predict(file_path)
            total_files += 1
            if actual_category == predicted_category:
                correct_predictions += 1
            print(f"File: {file}, Actual: {actual_category}, Predicted: {predicted_category} (Confidence Level: {confidence:.2f}%)")

accuracy = (correct_predictions / total_files) * 100
print(f"Final Accuracy: {accuracy:.2f}%")