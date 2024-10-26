
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify
import random
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

count = 0

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
app = Flask(__name__)

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route used to update counter variable
@app.route('/update_counter')
def update_counter():
    global count
    count += 1
    return jsonify(counter=count)

# Function to generate video frames
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam
    
    # Initialize the pose model 
    with mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.6) as pose:

        while True:
            success, frame = cap.read()  # Read the frame from webcam
            if not success:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Correct the color conversion

            if results.pose_landmarks:  # Check if landmarks are detected
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))  # Optional: Different color for connections

            ret, buffer = cv2.imencode('.jpg', image)  # Encode the frame in JPEG format
            image = buffer.tobytes()  # Convert to bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)

