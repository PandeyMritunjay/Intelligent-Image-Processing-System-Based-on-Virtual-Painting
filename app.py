import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time

# Streamlit Page Configuration
st.set_page_config(page_title="Virtual Painter", layout="wide")

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

# Session State Initialization
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# Welcome Page
if st.session_state.page == "welcome":
    st.markdown(
        """
        <style>
            body {
                background: linear-gradient(120deg, #f6d365, #fda085);
                font-family: 'Arial', sans-serif;
                color: white;
                text-align: center;
                margin: 0;
                padding: 0;
            }
            .welcome-container {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .welcome-header {
                font-size: 3rem;
                font-weight: bold;
                text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            }
            .welcome-subheader {
                font-size: 1.5rem;
                margin: 20px 0;
            }
            .welcome-button {
                background-color: #ff69b4;
                color: white;
                padding: 15px 30px;
                font-size: 1.2rem;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: transform 0.2s ease-in-out;
            }
            .welcome-button:hover {
                transform: scale(1.1);
                background-color: #ff85c4;
            }
        </style>
        <div class="welcome-container">
            <div class="welcome-header">🎨 Welcome to Virtual Painter!</div>
            <div class="welcome-subheader">Create art with your hands and let your creativity flow ✋🎨</div>
            <button class="welcome-button" onclick="startDrawing()">Start Drawing</button>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start Drawing 🎨"):
        st.session_state.page = "main"
    st.stop()

# Main Page
if st.session_state.page == "main":
    st.title("🎨 Virtual Painter with Gesture Controls")
    FRAME_WINDOW = st.image([])

    # Button to start/stop video capturing
    if "run" not in st.session_state:
        st.session_state.run = True

    def toggle_run():
        st.session_state.run = not st.session_state.run

    st.button("Start/Stop Video", on_click=toggle_run)

    # Initialize Webcam
    cap = cv2.VideoCapture(0)

    # Main Application Loop
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access the webcam. Please check your settings.")
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally
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

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()
