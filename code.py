import cv2
import numpy as np
import time
from pymycobot.mycobot import MyCobot

color_ranges = {
    "Red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([160, 100, 100]), np.array([180, 255, 255]))
    ],
    "Green": [(np.array([36, 100, 100]), np.array([85, 255, 255]))],
    "Blue": [(np.array([94, 80, 2]), np.array([126, 255, 255]))]
}

mc = MyCobot('/dev/ttyAMA0', 1000000)
cap = cv2.VideoCapture(0)

last_detected_color = None
last_detection_time = 0
detection_cooldown = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected = False

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
                current_time = time.time()

                # Only send movement if not recently triggered
                if (last_detected_color != color_name or
                        current_time - last_detection_time > detection_cooldown):
                    
                    print(f"Detected: {color_name}, moving robot.")
                    mc.send_angles([0, 0, 0, 0, 0, 0], 30)
                    time.sleep(1.5)

                    mc.send_angles([10, 0, -90, 10, 0, 0], 30)
                    time.sleep(1.5)

                    mc.send_angles([110, 10, -110, 15, 0, 0], 30)
                    time.sleep(1.5)

                    last_detected_color = color_name
                    last_detection_time = current_time
                break  # Stop checking other colors once one is found

    if not detected:
        last_detected_color = None  # Reset if nothing is detected

    cv2.imshow("Color Cube Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
