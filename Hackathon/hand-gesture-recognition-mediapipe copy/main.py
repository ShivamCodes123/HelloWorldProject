

"""
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import time

import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
app = Flask(__name__,  template_folder='/Users/aidendrep/Downloads/Hackathon/hand-gesture-recognition-mediapipe copy/templates')

count = -1

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('/Users/aidendrep/Downloads/Hackathon/hand-gesture-recognition-mediapipe copy/templates/index.html')

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route used to update counter variable
@app.route('/update_counter')
def update_counter():
    global count
    return jsonify(counter=count)

# Function to generate video frames
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam
    
    # Initialize the pose model 
    with mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.6) as pose:

            keypoint_classifier = KeyPointClassifier()
            point_history_classifier = PointHistoryClassifier()

                # Read labels ###########################################################
            with open('/Users/aidendrep/Downloads/Hackathon/hand-gesture-recognition-mediapipe copy/model/keypoint_classifier/keypoint_classifier_label.csv',
                        encoding='utf-8-sig') as f:
                keypoint_classifier_labels = csv.reader(f)
                keypoint_classifier_labels = [
                    row[0] for row in keypoint_classifier_labels
                ]
            with open('/Users/aidendrep/Downloads/Hackathon/hand-gesture-recognition-mediapipe copy/model/point_history_classifier/point_history_classifier_label.csv',
                    encoding='utf-8-sig') as f:
                point_history_classifier_labels = csv.reader(f)
                point_history_classifier_labels = [
                    row[0] for row in point_history_classifier_labels
                ]
            
            history_length = 16
            point_history = deque(maxlen=history_length)

            finger_gesture_history = deque(maxlen=history_length)

            cvFpsCalc = CvFpsCalc(buffer_len=10)

            

            mode = 0
            while cap.isOpened():
                fps = cvFpsCalc.get()
                key = cv.waitKey(10)
                if key == 27:  # ESC
                    break

                number, mode = select_mode(key, mode)
                ret, image = cap.read()

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Correct the color conversion
                debug_image = copy.deepcopy(image)
                
                if results.pose_landmarks:  # Check if landmarks are detected
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=10, circle_radius=6),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5, circle_radius=3))  # Optional: Different color for connections
                
                if results.pose_landmarks:  # Check if landmarks are detected
                    for pose_landmarks in (results.pose_landmarks.landmark):

                        brect = calc_bounding_rect(debug_image, pose_landmarks)

                        landmark_list = calc_landmark_list(debug_image, results.pose_landmarks.landmark)

                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                        # Hand sign classification
                        pose_sign_id = keypoint_classifier(pre_processed_landmark_list)
                        if pose_sign_id == 2:  # Point gesture
                            point_history.append(landmark_list[8])
                        else:
                            point_history.append([0, 0])

                        # Finger gesture classification
                        finger_gesture_id = 0
                        point_history_len = len(pre_processed_point_history_list)
                        if point_history_len == (history_length * 2):
                            finger_gesture_id = point_history_classifier(
                                pre_processed_point_history_list)

                        # Calculates the gesture IDs in the latest detection
                        finger_gesture_history.append(finger_gesture_id)
                        most_common_fg_id = Counter(
                            finger_gesture_history).most_common()

                        # Drawing part

                        image = draw_landmarks(debug_image, landmark_list)
                        image = draw_info_text(
                            image,
                            brect,
                            keypoint_classifier_labels[pose_sign_id],
                            point_history_classifier_labels[most_common_fg_id[0][0]]
                        )

                        ret, buffer = cv2.imencode('.jpg', image)  # Encode the frame in JPEG format
                        image = buffer.tobytes()  # Convert to bytes
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

                        print(count)
                        if keypoint_classifier_labels[pose_sign_id] != "Up" and keypoint_classifier_labels[pose_sign_id] != "Not In Position" and keypoint_classifier_labels[pose_sign_id] != "Butt to High" and keypoint_classifier_labels[pose_sign_id] != "Knees too low"  and not is_down:  
                            is_down = True
                        elif keypoint_classifier_labels[pose_sign_id] == "Up" and is_down:  # Only count if we were previously down
                            time.sleep(0.005)
                            count += 1
                            is_down = False
            
                        logging_csv(number, mode, pre_processed_landmark_list,
                                    pre_processed_point_history_list)
                        
                        pose_sign_id = keypoint_classifier(pre_processed_landmark_list)
                        if pose_sign_id == 2:  # Point gesture
                            point_history.append(landmark_list[8])
                        else:
                            point_history.append([0, 0])

                        finger_gesture_id = 0
                        point_history_len = len(pre_processed_point_history_list)
                        if point_history_len == (history_length * 2):
                            finger_gesture_id = point_history_classifier(
                                pre_processed_point_history_list)

                        # Calculates the gesture IDs in the latest detection
                        finger_gesture_history.append(finger_gesture_id)
                        most_common_fg_id = Counter(
                            finger_gesture_history).most_common()
                        
                    else:
                        point_history.append([0, 0])

                    debug_image = draw_point_history(debug_image, point_history)
                    debug_image = draw_info(debug_image, fps, mode, number)


            

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)


    
    





is_down = False


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args
    
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value


    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def draw_landmarks(image, landmark_point):
   if len(landmark_point) > 0:


       # right arm (actuall right)
       cv.line(image, tuple(landmark_point[12]), tuple(landmark_point[14]),
               (0, 0, 0), 6)
       cv.line(image, tuple(landmark_point[12]), tuple(landmark_point[14]),
               (255, 255, 255), 2)
       cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[16]),
               (0, 0, 0), 6)
       cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[16]),
               (255, 255, 255), 2)
      
       # right torso
       cv.line(image, tuple(landmark_point[12]), tuple(landmark_point[24]),
               (0, 0, 0), 6)
       cv.line(image, tuple(landmark_point[12]), tuple(landmark_point[24]),
               (255, 255, 255), 2)
      
       #right leg
       cv.line(image, tuple(landmark_point[24]), tuple(landmark_point[26]),
               (0, 0, 0), 6)
       cv.line(image, tuple(landmark_point[24]), tuple(landmark_point[26]),
               (255, 255, 255), 2)
       cv.line(image, tuple(landmark_point[26]), tuple(landmark_point[28]),
               (0, 0, 0), 6)
       cv.line(image, tuple(landmark_point[26]), tuple(landmark_point[28]),
               (255, 255, 255), 2)
      


       # left arm (actuall right)
       cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[13]),
               (0, 0, 0), 6)
       cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[13]),
               (255, 255, 255), 2)
       cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[15]),
               (0, 0, 0), 6)
       cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[15]),
               (255, 255, 255), 2)
      
       # left torso
       cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[23]),
               (0, 0, 0), 6)
       cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[23]),
               (255, 255, 255), 2)
      
       #left leg
       cv.line(image, tuple(landmark_point[23]), tuple(landmark_point[25]),
               (0, 0, 0), 6)
       cv.line(image, tuple(landmark_point[23]), tuple(landmark_point[25]),
               (255, 255, 255), 2)
       cv.line(image, tuple(landmark_point[25]), tuple(landmark_point[27]),
               (0, 0, 0), 6)
       cv.line(image, tuple(landmark_point[25]), tuple(landmark_point[27]),
               (255, 255, 255), 2)
       
       for index, landmark in enumerate(landmark_point):
        if index == 11:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 13:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 16:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 23:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 24:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 25:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 26:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 27:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 28:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

        return image


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = '/Users/aidendrep/Downloads/Hackathon/hand-gesture-recognition-mediapipe copy/model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = '/Users/aidendrep/Downloads/Hackathon/hand-gesture-recognition-mediapipe copy/model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                    (152, 251, 152), 2)

    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(results.pose_landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_info_text(image, brect, pose_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = ""
    if pose_sign_text != "":
        info_text = pose_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv.LINE_AA)
    return image
"""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import time


import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque


import cv2 as cv
import numpy as np


from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier



mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils



cap = cv2.VideoCapture(0)
is_down = False




def get_args():
   parser = argparse.ArgumentParser()


   parser.add_argument("--device", type=int, default=0)
   parser.add_argument("--width", help='cap width', type=int, default=960)
   parser.add_argument("--height", help='cap height', type=int, default=540)


   parser.add_argument('--use_static_image_mode', action='store_true')
   parser.add_argument("--min_detection_confidence",
                       help='min_detection_confidence',
                       type=float,
                       default=0.7)
   parser.add_argument("--min_tracking_confidence",
                       help='min_tracking_confidence',
                       type=int,
                       default=0.5)


   args = parser.parse_args()


   return args
  
def calc_landmark_list(image, landmarks):
   image_width, image_height = image.shape[1], image.shape[0]


   landmark_point = []


   # Keypoint
   for _, landmark in enumerate(landmarks):
       landmark_x = min(int(landmark.x * image_width), image_width - 1)
       landmark_y = min(int(landmark.y * image_height), image_height - 1)
       # landmark_z = landmark.z


       landmark_point.append([landmark_x, landmark_y])


   return landmark_point


def pre_process_landmark(landmark_list):
   temp_landmark_list = copy.deepcopy(landmark_list)


   # Convert to relative coordinates
   base_x, base_y = 0, 0
   for index, landmark_point in enumerate(temp_landmark_list):
       if index == 0:
           base_x, base_y = landmark_point[0], landmark_point[1]


       temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
       temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y


   # Convert to a one-dimensional list
   temp_landmark_list = list(
       itertools.chain.from_iterable(temp_landmark_list))


   # Normalization
   max_value = max(list(map(abs, temp_landmark_list)))


   def normalize_(n):
       return n / max_value




   temp_landmark_list = list(map(normalize_, temp_landmark_list))


   return temp_landmark_list




def pre_process_point_history(image, point_history):
   image_width, image_height = image.shape[1], image.shape[0]


   temp_point_history = copy.deepcopy(point_history)


   # Convert to relative coordinates
   base_x, base_y = 0, 0
   for index, point in enumerate(temp_point_history):
       if index == 0:
           base_x, base_y = point[0], point[1]


       temp_point_history[index][0] = (temp_point_history[index][0] -
                                       base_x) / image_width
       temp_point_history[index][1] = (temp_point_history[index][1] -
                                       base_y) / image_height


   # Convert to a one-dimensional list
   temp_point_history = list(
       itertools.chain.from_iterable(temp_point_history))


   return temp_point_history


def select_mode(key, mode):
   number = -1
   if 48 <= key <= 57:  # 0 ~ 9
       number = key - 48
   if key == 110:  # n
       mode = 0
   if key == 107:  # k
       mode = 1
   if key == 104:  # h
       mode = 2
   return number, mode


def draw_landmarks(image, landmark_point):
  if len(landmark_point) > 0:




      # right arm (actuall right)
      cv.line(image, tuple(landmark_point[12]), tuple(landmark_point[14]),
              (0, 0, 0), 6)
      cv.line(image, tuple(landmark_point[12]), tuple(landmark_point[14]),
              (255, 255, 255), 2)
      cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[16]),
              (0, 0, 0), 6)
      cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[16]),
              (255, 255, 255), 2)
    
      # right torso
      cv.line(image, tuple(landmark_point[12]), tuple(landmark_point[24]),
              (0, 0, 0), 6)
      cv.line(image, tuple(landmark_point[12]), tuple(landmark_point[24]),
              (255, 255, 255), 2)
    
      #right leg
      cv.line(image, tuple(landmark_point[24]), tuple(landmark_point[26]),
              (0, 0, 0), 6)
      cv.line(image, tuple(landmark_point[24]), tuple(landmark_point[26]),
              (255, 255, 255), 2)
      cv.line(image, tuple(landmark_point[26]), tuple(landmark_point[28]),
              (0, 0, 0), 6)
      cv.line(image, tuple(landmark_point[26]), tuple(landmark_point[28]),
              (255, 255, 255), 2)
    




      # left arm (actuall right)
      cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[13]),
              (0, 0, 0), 6)
      cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[13]),
              (255, 255, 255), 2)
      cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[15]),
              (0, 0, 0), 6)
      cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[15]),
              (255, 255, 255), 2)
    
      # left torso
      cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[23]),
              (0, 0, 0), 6)
      cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[23]),
              (255, 255, 255), 2)
    
      #left leg
      cv.line(image, tuple(landmark_point[23]), tuple(landmark_point[25]),
              (0, 0, 0), 6)
      cv.line(image, tuple(landmark_point[23]), tuple(landmark_point[25]),
              (255, 255, 255), 2)
      cv.line(image, tuple(landmark_point[25]), tuple(landmark_point[27]),
              (0, 0, 0), 6)
      cv.line(image, tuple(landmark_point[25]), tuple(landmark_point[27]),
              (255, 255, 255), 2)
     
      for index, landmark in enumerate(landmark_point):
       if index == 11:  # 手首1
           cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
           cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
       if index == 12:  # 手首2
           cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
           cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
       if index == 13:  # 親指：付け根
           cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
           cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
       if index == 14:  # 親指：第1関節
           cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
           cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
       if index == 15:  # 親指：指先
           cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                       -1)
           cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
       if index == 16:  # 人差指：付け根
           cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
           cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
       if index == 23:  # 人差指：第2関節
           cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
           cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
       if index == 24:  # 人差指：第1関節
           cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
           cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
       if index == 25:  # 人差指：指先
           cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                       -1)
           cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
       if index == 26:  # 中指：付け根
           cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
           cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
       if index == 27:  # 中指：第2関節
           cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
           cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
       if index == 28:  # 中指：第1関節
           cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
           cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)


       return image




def logging_csv(number, mode, landmark_list, point_history_list):
   if mode == 0:
       pass
   if mode == 1 and (0 <= number <= 9):
       csv_path = '/Users/aidendrep/Downloads/Hackathon/hand-gesture-recognition-mediapipe copy/model/keypoint_classifier/keypoint.csv'
       with open(csv_path, 'a', newline="") as f:
           writer = csv.writer(f)
           writer.writerow([number, *landmark_list])
   if mode == 2 and (0 <= number <= 9):
       csv_path = '/Users/aidendrep/Downloads/Hackathon/hand-gesture-recognition-mediapipe copy/model/point_history_classifier/point_history.csv'
       with open(csv_path, 'a', newline="") as f:
           writer = csv.writer(f)
           writer.writerow([number, *point_history_list])
   return


def draw_point_history(image, point_history):
   for index, point in enumerate(point_history):
       if point[0] != 0 and point[1] != 0:
           cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                   (152, 251, 152), 2)


   return image


def calc_bounding_rect(image, landmarks):
   image_width, image_height = image.shape[1], image.shape[0]


   landmark_array = np.empty((0, 2), int)


   for _, landmark in enumerate(results.pose_landmarks.landmark):
       landmark_x = min(int(landmark.x * image_width), image_width - 1)
       landmark_y = min(int(landmark.y * image_height), image_height - 1)


       landmark_point = [np.array((landmark_x, landmark_y))]


       landmark_array = np.append(landmark_array, landmark_point, axis=0)


   x, y, w, h = cv.boundingRect(landmark_array)


   return [x, y, x + w, y + h]




def draw_info_text(image, brect, pose_sign_text,
                  finger_gesture_text):
   cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                (0, 0, 0), -1)


   info_text = ""
   if pose_sign_text != "":
       info_text = pose_sign_text
   cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)


   if finger_gesture_text != "":
       cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
       cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                  cv.LINE_AA)


   return image




def draw_info(image, fps, mode, number, count):
   cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
           1.0, (0, 0, 0), 4, cv.LINE_AA)
   cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
           1.0, (255, 255, 255), 2, cv.LINE_AA)
   cv.putText(image, "Push Up Count: " + str(count + 1), (70, 100), cv.FONT_HERSHEY_SIMPLEX,
              2.0, (255, 255, 255), 6, cv.LINE_AA)


   mode_string = ['Logging Key Point', 'Logging Point History']
   if 1 <= mode <= 2:
       cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
               cv.LINE_AA)
       if 0 <= number <= 9:
           cv.putText(image, "NUM:" + str(number), (10, 110),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
   return image


with mp_pose.Pose(
       model_complexity=0,
       min_detection_confidence=0.4,
       min_tracking_confidence=0.6) as pose:
  
   keypoint_classifier = KeyPointClassifier()
   point_history_classifier = PointHistoryClassifier()


       # Read labels ###########################################################
   with open('/Users/aidendrep/Downloads/Hackathon/hand-gesture-recognition-mediapipe copy/model/keypoint_classifier/keypoint_classifier_label.csv',
               encoding='utf-8-sig') as f:
       keypoint_classifier_labels = csv.reader(f)
       keypoint_classifier_labels = [
           row[0] for row in keypoint_classifier_labels
       ]
   with open('/Users/aidendrep/Downloads/Hackathon/hand-gesture-recognition-mediapipe copy/model/point_history_classifier/point_history_classifier_label.csv',
           encoding='utf-8-sig') as f:
       point_history_classifier_labels = csv.reader(f)
       point_history_classifier_labels = [
           row[0] for row in point_history_classifier_labels
       ]
  
   history_length = 16
   point_history = deque(maxlen=history_length)


   finger_gesture_history = deque(maxlen=history_length)


   cvFpsCalc = CvFpsCalc(buffer_len=10)


  


   mode = 0
   count = -1
   while cap.isOpened():
       fps = cvFpsCalc.get()
       key = cv.waitKey(10)
       if key == 27:  # ESC
           break


       number, mode = select_mode(key, mode)
       ret, image = cap.read()


       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       image.flags.writeable = False


       results = pose.process(image)
       image.flags.writeable = True
       image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Correct the color conversion
       debug_image = copy.deepcopy(image)
      
       if results.pose_landmarks:  # Check if landmarks are detected
           mp_drawing.draw_landmarks(
               image,
               results.pose_landmarks,
               mp_pose.POSE_CONNECTIONS,
               mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=10, circle_radius=6),
               mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5, circle_radius=3))  # Optional: Different color for connections
      
       if results.pose_landmarks:  # Check if landmarks are detected
           for pose_landmarks in (results.pose_landmarks.landmark):


               brect = calc_bounding_rect(debug_image, pose_landmarks)


               landmark_list = calc_landmark_list(debug_image, results.pose_landmarks.landmark)


               pre_processed_landmark_list = pre_process_landmark(landmark_list)
               pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)


               # Hand sign classification
               pose_sign_id = keypoint_classifier(pre_processed_landmark_list)
               if pose_sign_id == 2:  # Point gesture
                   point_history.append(landmark_list[8])
               else:
                   point_history.append([0, 0])


               # Finger gesture classification
               finger_gesture_id = 0
               point_history_len = len(pre_processed_point_history_list)
               if point_history_len == (history_length * 2):
                   finger_gesture_id = point_history_classifier(
                       pre_processed_point_history_list)


               # Calculates the gesture IDs in the latest detection
               finger_gesture_history.append(finger_gesture_id)
               most_common_fg_id = Counter(
                   finger_gesture_history).most_common()


               # Drawing part


               image = draw_landmarks(debug_image, landmark_list)
               image = draw_info_text(
                   image,
                   brect,
                   keypoint_classifier_labels[pose_sign_id],
                   point_history_classifier_labels[most_common_fg_id[0][0]]
               )


               print(count)
               #if keypoint_classifier_labels[pose_sign_id] != "Up" and keypoint_classifier_labels[pose_sign_id] != "Not In Position" and keypoint_classifier_labels[pose_sign_id] != "Butt to High" and keypoint_classifier_labels[pose_sign_id] != "Knees too low"  and not is_down: 
               if keypoint_classifier_labels[pose_sign_id] == "Down":
                   is_down = True
               elif keypoint_classifier_labels[pose_sign_id] == "Up" and is_down:  # Only count if we were previously down
                   time.sleep(0.005)
                   count += 1
                   is_down = False
  
               logging_csv(number, mode, pre_processed_landmark_list,
                           pre_processed_point_history_list)
              
               pose_sign_id = keypoint_classifier(pre_processed_landmark_list)
               if pose_sign_id == 2:  # Point gesture
                   point_history.append(landmark_list[8])
               else:
                   point_history.append([0, 0])


               finger_gesture_id = 0
               point_history_len = len(pre_processed_point_history_list)
               if point_history_len == (history_length * 2):
                   finger_gesture_id = point_history_classifier(
                       pre_processed_point_history_list)


               # Calculates the gesture IDs in the latest detection
               finger_gesture_history.append(finger_gesture_id)
               most_common_fg_id = Counter(
                   finger_gesture_history).most_common()
              
           else:
               point_history.append([0, 0])


           debug_image = draw_point_history(debug_image, point_history)
           debug_image = draw_info(debug_image, fps, mode, number, count)




      




       cv2.imshow('Mediapipe feed', image)  # Use the processed image for display


       if cv2.waitKey(10) & 0xFF == ord('q'):
           break


      


cap.release()
cv2.destroyAllWindows()



