import cv2
import numpy as np
import threading
import time
from pymycobot.mycobot import MyCobot

# HSV color ranges
color_ranges = {
    "Red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([160, 100, 100]), np.array([180, 255, 255]))
    ],
    "Green": [(np.array([36, 100, 100]), np.array([85, 255, 255]))],
    "Blue": [(np.array([94, 80, 2]), np.array([126, 255, 255]))]
}

# Robot setup
mc = MyCobot('/dev/ttyAMA0', 1000000)

# Control flag
moving = False
last_color = None
cooldown = 3  # seconds

def move_robot(color_name):
    global moving, last_color
    print(f"Started movement for {color_name}")
    moving = True

    mc.send_angles([0, 0, 0, 0, 0, 0], 30)
    time.sleep(1.2)
    mc.send_angles([10, 0, -90, 10, 0, 0], 30)
    time.sleep(1.2)
    mc.send_angles([110, 10, -110, 15, 0, 0], 30)
    time.sleep(1.2)

    last_color = color_name
    moving = False
    print(f"Completed movement for {color_name}")

# Camera setup
cap = cv2.VideoCapture(0)
cv2.namedWindow("Color Cube Detection")

last_detection_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected = False
    current_time = time.time()

    for color_name, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            if mask is None:
                mask = cv2.inRange(hsv, lower, upper)
            else:
                mask += cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 800:
                x, y, w, h = cv2.boundingRect(largest)
                cx, cy = x + w // 2, y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, color_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                detected = True

                if (not moving and 
                    (last_color != color_name or (current_time - last_detection_time) > cooldown)):

                    last_detection_time = current_time
                    # Start movement in background thread
                    threading.Thread(target=move_robot, args=(color_name,), daemon=True).start()

                break  # stop checking other colors once one is found

    if not detected:
        last_color = None

    cv2.imshow("Color Cube Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
