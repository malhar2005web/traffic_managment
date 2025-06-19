import cv2
import numpy as np
import time
import tensorflow as tf

# Load TensorFlow model
model = tf.keras.models.load_model('traffic_tf_model.h5')

def count_vehicles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    vehicle_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            vehicle_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    return frame, vehicle_count

def predict_green_time(vehicle_count):
    input_val = np.array([[vehicle_count]], dtype=float)
    prediction = model.predict(input_val, verbose=0)
    green_time = min(max(int(prediction[0][0]), 5), 25)
    return green_time

cap = cv2.VideoCapture('traffic.mp4')  # or 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, vehicle_count = count_vehicles(frame)
    green_time = predict_green_time(vehicle_count)

    cv2.putText(frame, f'Vehicles: {vehicle_count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f'Green Light Duration: {green_time}s', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow('Smart TF Traffic System', frame)

    time.sleep(green_time)  # Simulate green signal time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()