from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
import os

app = Flask(__name__, template_folder="templates")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1)

# Drawing settings
width, height = 640, 480
imgCanvas = np.zeros((height, width, 3), np.uint8)
drawColor = (0, 0, 255)  # Default color: red
thickness = 15
xp, yp = 0, 0
tipIds = [4, 8, 12, 16, 20]  # Tip IDs of fingers: Thumb, Index, Middle, Ring, Pinky

@app.route('/')
def welcome():
    """Serve the welcome page."""
    return render_template('welcome.html')

@app.route('/main')
def index():
    """Serve the main virtual painter page."""
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process the video frame for hand tracking and drawing."""
    global xp, yp, imgCanvas, drawColor, thickness
    data = request.json
    frame_data = data['frame']

    # Decode the image from base64
    encoded_data = frame_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)

    # Process the frame for hand tracking
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get hand keypoints
            points = []
            for lm in hand_landmarks.landmark:
                points.append([int(lm.x * width), int(lm.y * height)])

            if len(points) != 0:
                x1, y1 = points[8]  # Index finger tip
                x2, y2 = points[12]  # Middle finger tip
                x4, y4 = points[4]  # Thumb tip

                # Check which fingers are up
                fingers = []
                for id in range(5):
                    if id == 0:  # Thumb check (left/right)
                        if points[tipIds[id]][0] < points[tipIds[id] - 1][0]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    else:  # Other fingers check (up/down)
                        if points[tipIds[id]][1] < points[tipIds[id] - 2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                # Gesture-Based Functionalities
                # 1. Selection mode (two fingers up)
                if fingers[1] and fingers[2] and not any(fingers[3:]):
                    xp, yp = 0, 0  # Reset drawing position
                    if y1 < 50:  # Assume menu bar is at the top
                        if 0 < x1 < 150:
                            drawColor = (0, 0, 255)  # Red
                        elif 150 < x1 < 300:
                            drawColor = (255, 0, 0)  # Blue
                        elif 300 < x1 < 450:
                            drawColor = (0, 255, 0)  # Green
                        elif 450 < x1 < 600:
                            drawColor = (0, 0, 0)  # Black
                        elif 600 < x1 < 750:
                            imgCanvas = np.zeros((height, width, 3), np.uint8)  # Clear canvas

                # 2. Draw mode (only index finger up)
                if fingers[1] and not any(fingers[2:]):
                    cv2.circle(frame, (x1, y1), 15, drawColor, cv2.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                    xp, yp = x1, y1
                else:
                    xp, yp = 0, 0

                # 3. Erase mode (all fingers down)
                if all(f == 0 for f in fingers):
                    imgCanvas = np.zeros((height, width, 3), np.uint8)

                # 4. Adjust brush thickness using index finger and thumb
                if fingers[1] and fingers[0] and not any(fingers[2:]):
                    distance = int(np.sqrt((x1 - x4)**2 + (y1 - y4)**2))
                    thickness = max(5, min(50, distance // 2))

    # Combine the frame with the canvas
    gray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, imgCanvas)

    # Encode the processed frame back to base64
    _, buffer = cv2.imencode('.jpg', frame)
    processed_frame = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"processed_frame": processed_frame})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
