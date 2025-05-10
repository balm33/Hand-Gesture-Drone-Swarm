import cv2
import os
import time 

# simple script to take and save images for testing purpose

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture (Press SPACE to save)", frame)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):  # ESC key
        break
    elif key == 32:  # SPACE key
        timestamp = time.time()
        filename = os.path.join(SAVE_DIR, f"img_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

cap.release()
cv2.destroyAllWindows()

