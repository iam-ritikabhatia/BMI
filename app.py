from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import os
import math

app = Flask(__name__)

# Load the Haar cascade file for face detection
opencv_dir = os.path.dirname(cv2.__file__)
face_cascade = cv2.CascadeClassifier(os.path.join(opencv_dir, 'data', 'haarcascade_frontalface_default.xml'))

cap = cv2.VideoCapture(0)

def extract_features(face_gray):
    face_array = np.array(face_gray)
    eye_distance = calculate_eye_distance(face_array)
    return {'eye_distance': eye_distance}

def calculate_eye_distance(face_array):
    _, thresh = cv2.threshold(face_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    eye_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if area > 100 and aspect_ratio > 2:
            eye_contours.append(contour)
    if len(eye_contours) == 2:
        x1, y1, w1, h1 = cv2.boundingRect(eye_contours[0])
        x2, y2, w2, h2 = cv2.boundingRect(eye_contours[1])
        eye_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return eye_distance
    else:
        return 0

def estimate_bmi(face_features):
    bmi = 20 + face_features['eye_distance'] * 0.1
    return bmi

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_gray = gray[y:y+h, x:x+w]
            face_features = extract_features(face_gray)
            bmi = estimate_bmi(face_features)
            cv2.putText(frame, f'BMI: {bmi:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bmi')
def bmi():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_features = extract_features(face_gray)
            bmi_value = estimate_bmi(face_features)
            return jsonify(bmi=bmi_value)
            
    else:
        return jsonify(bmi=0)

if __name__ == '__main__':
    app.run(debug=True)
