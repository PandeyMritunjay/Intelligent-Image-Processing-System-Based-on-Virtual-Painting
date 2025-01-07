from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
import os
import math

app = Flask(__name__, template_folder="templates")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1)

# Drawing settings
width, height = 1280, 720
imgCanvas = np.zeros((height, width, 3), np.uint8)
drawColor = (0, 0, 255)
thickness = 15
xp, yp = 0, 0
tipIds = [4, 8, 12, 16, 20]  # Fingertips indexes

# Load header images for color selection
myList = [file for file in os.listdir('./static') if file.endswith('.png')]
overlayList = [cv2.imread(f'./static/{imPath}') for imPath in myList]
header = overlayList[0]

@app.route('/')
def index():
    """Renders the homepage with webcam integration."""
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Processes video frames received from the frontend."""
    global xp, yp, imgCanvas, drawColor, header, thickness
    data = request.json
    frame_data = data['frame']

    # Decode the image from base64
    encoded_data = frame_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Flip and process frame
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Check for hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get fingertip positions
            points = []
            for lm in hand_landmarks.landmark:
                points.append([int(lm.x * width), int(lm.y * height)])

            if len(points) != 0:
                x1, y1 = points[8]  # Index finger
                x2, y2 = points[12]  # Middle finger

                # Check which fingers are up
                fingers = []
                if points[tipIds[0]][0] < points[tipIds[0] - 1][0]:
                    fingers.append(1)  # Thumb
                else:
                    fingers.append(0)

                for id in range(1, 5):
                    if points[tipIds[id]][1] < points[tipIds[id] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Selection mode (two fingers up)
                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0
                    if y1 < 125:  # Selecting header color
                        if 170 < x1 < 295:
                            header = overlayList[0]
                            drawColor = (0, 0, 255)
                        elif 436 < x1 < 561:
                            header = overlayList[1]
                            drawColor = (255, 0, 0)
                        elif 700 < x1 < 825:
                            header = overlayList[2]
                            drawColor = (0, 255, 0)
                        elif 980 < x1 < 1105:
                            header = overlayList[3]
                            drawColor = (0, 0, 0)
                    cv2.rectangle(frame, (x1 - 10, y1 - 15), (x2 + 10, y2 + 23), drawColor, cv2.FILLED)

                # Draw mode (only index finger up)
                if fingers[1] and not any(fingers[2:]):
                    cv2.circle(frame, (x1, y1), 15, drawColor, cv2.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                    xp, yp = x1, y1

                # Erase mode (all fingers down)
                if all(f == 0 for f in fingers):
                    imgCanvas = np.zeros((height, width, 3), np.uint8)

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
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
