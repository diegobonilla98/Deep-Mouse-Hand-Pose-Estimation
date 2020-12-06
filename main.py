import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from pymouse import PyMouse


def distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def get_screen_resolution():
    output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4', shell=True, stdout=subprocess.PIPE).communicate()[0]
    resolution = output.split()[0].split(b'x')
    return int(resolution[0]), int(resolution[1])


def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

m = PyMouse()

wait_time = 0
WAITING = False

BUFFER_SIZE = 10
BUFFER_IDX = 0
BUFFER_TOTAL_X = 0
BUFFER_TOTAL_Y = 0
buffer_x_coords = [0] * BUFFER_SIZE
buffer_y_coords = [0] * BUFFER_SIZE

prev_pos = None
SCROLLING = False
DEBUG_MODE = True
SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_resolution()
NOT_SIGNAL = 0

cam = cv2.VideoCapture(0)
while True:
    if NOT_SIGNAL > 30:
        break
    ret, frame = cam.read()
    image_rows, image_cols = frame.shape[:2]

    results = hands.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1))
    if not results.multi_hand_landmarks:
        NOT_SIGNAL += 1
        continue
    annotated_image = cv2.flip(frame.copy(), 1)
    for hand_landmarks in results.multi_hand_landmarks:
        landmarks = [mp_drawing._normalized_to_pixel_coordinates(l.x, l.y, image_cols, image_rows) for l in
                     hand_landmarks.landmark]
        if not all(landmarks):
            NOT_SIGNAL += 1
            continue
        hand_w = max(landmarks, key=lambda x: x[0])[0] - min(landmarks, key=lambda x: x[0])[0]
        hand_h = max(landmarks, key=lambda x: x[1])[1] - min(landmarks, key=lambda x: x[1])[1]

        d = distance((landmarks[8][0] / hand_w, landmarks[8][1] / hand_h), (landmarks[12][0] / hand_w, landmarks[12][1] / hand_h)) < 0.25
        if not SCROLLING and d:
            SCROLLING = True
            prev_pos = landmarks[8]
        if SCROLLING and d:
            m.scroll(horizontal=(landmarks[8][0] - prev_pos[0]) // 20, vertical=(landmarks[8][1] - prev_pos[1]) // 20)
        if not d:
            SCROLLING = False

        landmarks = np.array(landmarks)
        median_center = landmarks[4]

        mouse_x = int(translate(landmarks[8][0], 0, image_cols, 0, SCREEN_WIDTH))
        mouse_y = int(translate(landmarks[8][1], 0, image_rows, SCREEN_HEIGHT / 2, SCREEN_HEIGHT * 4))
        if wait_time > 5:
            if distance((landmarks[8][0] / hand_w, landmarks[8][1] / hand_h),
                        (median_center[0] / hand_w, median_center[1] / hand_h)) < 0.2:
                m.click(mouse_x, mouse_y, n=2)
            wait_time = 0
            WAITING = False
        if WAITING:
            wait_time += 1
        elif not SCROLLING:
            BUFFER_TOTAL_X -= buffer_x_coords[BUFFER_IDX]
            BUFFER_TOTAL_Y -= buffer_y_coords[BUFFER_IDX]

            buffer_x_coords[BUFFER_IDX] = mouse_x
            buffer_y_coords[BUFFER_IDX] = mouse_y

            BUFFER_TOTAL_X += buffer_x_coords[BUFFER_IDX]
            BUFFER_TOTAL_Y += buffer_y_coords[BUFFER_IDX]

            BUFFER_IDX += 1

            if BUFFER_IDX >= BUFFER_SIZE:
                BUFFER_IDX = 0

            mouse_x = BUFFER_TOTAL_X // BUFFER_SIZE
            mouse_y = BUFFER_TOTAL_Y // BUFFER_SIZE

            if distance((landmarks[8][0] / hand_w, landmarks[8][1] / hand_h), (median_center[0] / hand_w, median_center[1] / hand_h)) < 0.2:
                m.click(mouse_x, mouse_y)
                WAITING = True
            elif wait_time > 5 or wait_time == 0:
                wait_time = 0
                m.move(mouse_x, mouse_y)

        if DEBUG_MODE:
            cv2.circle(annotated_image, (int(median_center[0]), int(median_center[1])), 20, (0, 255, 0), -1)
            for idx, landmark in enumerate(landmarks):
                if landmark is None:
                    continue
                cv2.circle(annotated_image, (landmark[0], landmark[1]), 10, (255, 0, 0), -1)
                cv2.putText(annotated_image, str(idx), (landmark[0] + 20, landmark[1] + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (255, 0, 0), 1)
    cv2.imshow('Output', annotated_image)
    NOT_SIGNAL = 0
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
