import cv2
import numpy as np
import threading
import time
from pymycobot.mycobot import MyCobot
import RPi.GPIO as GPIO

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
GPIO.setmode(GPIO.BCM)
GPIO.setup(20, GPIO.OUT)
GPIO.setup(21, GPIO.OUT)

# Control flag
moving = False
last_color = None
cooldown = 3  # seconds

def move_robot(color_name):
    global moving, last_color
    print(f"Started movement for {color_name}")
    moving = True
    time.sleep(1)

    if color_name == "Blue":
        mc.send_angles([0, 0, 0, 0, 0, 0], 30)
        mc.send_angles([10, 0, -90, 10, 0, 0], 30)
        mc.send_angles([25, -20, -145, 80, 0, 0], 30)
        GPIO.output(20, GPIO.LOW)
        GPIO.output(21, GPIO.LOW)
        time.sleep(2)
        mc.send_angles([10, 0, -90, 10, 0, 0], 30)
        mc.send_angles([-20, -65, 0, -15, 0, 0], 30)
        mc.send_angles([-20, -70, -10, 0, 0, 0], 30)
        GPIO.output(20, GPIO.HIGH)
        GPIO.output(21, GPIO.HIGH)
        time.sleep(7)
        mc.send_angles([10, 0, -90, 10, 0, 0], 30)
        mc.send_angles([0, 0, 0, 0, 0, 0], 30)

    elif color_name == "Red":
        mc.send_angles([0, 0, 0, 0, 0, 0], 30)
        mc.send_angles([10, 0, -90, 10, 0, 0], 30)
        mc.send_angles([20, -25, -145, 85, 0, 0], 30)
        GPIO.output(20, GPIO.LOW)
        GPIO.output(21, GPIO.LOW)
        time.sleep(2)
        mc.send_angles([10, 0, -90, 10, 0, 0], 30)
        mc.send_angles([-30, -10, -100, 30, 0, 0], 30)
        mc.send_angles([-30, -20, -100, 35, 0, 0], 30)
        GPIO.output(20, GPIO.HIGH)
        GPIO.output(21, GPIO.HIGH)
        time.sleep(7)
        mc.send_angles([10, 0, -90, 10, 0, 0], 30)
        mc.send_angles([0, 0, 0, 0, 0, 0], 30)

    elif color_name == "Green":
        mc.send_angles([0, 0, 0, 0, 0, 0], 30)
        mc.send_angles([10, 0, -90, 10, 0, 0], 30)
        mc.send_angles([20, -25, -145, 85, 0, 0], 30)
        GPIO.output(20, GPIO.LOW)
        GPIO.output(21, GPIO.LOW)
        time.sleep(2)
        mc.send_angles([10, 0, -90, 10, 0, 0], 30)
        mc.send_angles([110, 10, -110, 15, 0, 0], 30)
        mc.send_angles([110, 0, -120, 35, 0, 0], 30)
        GPIO.output(20, GPIO.HIGH)
        GPIO.output(21, GPIO.HIGH)
        time.sleep(7)
        mc.send_angles([10, 0, -90, 10, 0, 0], 30)
        mc.send_angles([0, 0, 0, 0, 0, 0], 30)

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

    frame_height, frame_width = frame.shape[:2]

    # Define 13cm × 13cm detection zone in the center (adjust pixels as needed)
    box_size = 300  # in pixels (you can calibrate based on actual cm/pixels)
    start_x = frame_width // 2 - box_size // 2
    start_y = frame_height // 2 - box_size // 2
    end_x = start_x + box_size
    end_y = start_y + box_size

    # Draw detection zone
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 0), 2)
    cv2.putText(frame, "Detection Zone", (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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

                # Only allow detection inside the defined 13×13 cm zone
                if start_x <= cx <= end_x and start_y <= cy <= end_y:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, color_name, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    detected = True

                    if (not moving and
                        (last_color != color_name or (current_time - last_detection_time) > cooldown)):

                        last_detection_time = current_time
                        threading.Thread(target=move_robot, args=(color_name,), daemon=True).start()

                    break  # stop checking other colors once one is found

    if not detected:
        last_color = None

    cv2.imshow("Color Cube Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
