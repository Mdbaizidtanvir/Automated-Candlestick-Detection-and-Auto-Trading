import cv2
import numpy as np
from mss import mss
import keyboard
import time
import pyautogui
import os
from ultralytics import YOLO

# Initialize YOLO model with pre-trained weights
model = YOLO(r'E:\train8\train8\weights\best.pt')

# Define screen dimensions for capture
screen = {"top": 0, "left": 0, "width": 1920, "height": 1080}
sct = mss()

# Template directories
template_directory_higher = "./bi-images/higher/"
template_directory_lower = "./bi-images/lower/"
template_directory_skip = "./bi-images/skip/"

# Read template images from the directories
template_images_higher = [cv2.imread(os.path.join(template_directory_higher, filename)) for filename in os.listdir(template_directory_higher) if filename.endswith(".png")]
template_images_lower = [cv2.imread(os.path.join(template_directory_lower, filename)) for filename in os.listdir(template_directory_lower) if filename.endswith(".png")]
template_images_skip = [cv2.imread(os.path.join(template_directory_skip, filename)) for filename in os.listdir(template_directory_skip) if filename.endswith(".png")]

# Create a resizable window
cv2.namedWindow('Real-Time Object Recognition', cv2.WINDOW_NORMAL)

cooldown = 60  # Cooldown period in seconds

def detect_from_model(frame):
    results = model.predict(frame, conf=0.15)
    for result in results:
        if result.boxes:
            box = result.boxes[0]
            class_id = int(box.cls)
            object_name = model.names[class_id]
            if object_name == "buy":
                print("buy is found")
                if time.time() - main_app.last_click_time > cooldown:
                    click_x = 1781
                    click_y = 550
                    pyautogui.click(click_x, click_y)
                    main_app.last_click_time = time.time()
                    # Pressing Shift + D
                    pyautogui.keyDown('shift')
                    pyautogui.press('d')
                    pyautogui.keyUp('shift')
                # Draw green border around the object
                cv2.rectangle(frame, (int(box.xmin), int(box.ymin)), (int(box.xmax), int(box.ymax)), (0, 255, 0), 2)

            if object_name == "sell":
                print("sell is found")
                if time.time() - main_app.last_click_time > cooldown:
                    click_x = 1757
                    click_y = 572
                    pyautogui.click(click_x, click_y)
                    main_app.last_click_time = time.time()
                    # Pressing Shift + D
                    pyautogui.keyDown('shift')
                    pyautogui.press('d')
                    pyautogui.keyUp('shift')
                # Draw yellow border around the object
                cv2.rectangle(frame, (int(box.xmin), int(box.ymin)), (int(box.xmax), int(box.ymax)), (0, 255, 255), 2)
    return frame

def template_matching(frame):
    # Perform template matching here
    # This is just a placeholder, you need to implement the actual template matching logic
    # and draw rectangles on the frame accordingly
    return frame

def main_app():
    last_click_time = 0  # Initialize last_click_time
    global cooldown

    while True:
        try:
            # Screen capture and conversion for OpenCV processing
            sct_img = sct.grab(screen)
            frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

            # Detect objects using model
            frame_model = detect_from_model(frame.copy())

            # Perform template matching
            frame_template = template_matching(frame.copy())

            # Combine frames for display
            combined_frame = np.hstack((frame_model, frame_template))

            # Display processed frame in a resizable window
            cv2.imshow('Real-Time Object Recognition', combined_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):  # Exit on 'q'
                break

            if keyboard.is_pressed("a"):
                print("Exiting...")
                break

            if keyboard.is_pressed("s"): # For exit
                print("Exiting...")
                break
        except Exception as e:
            print("An error occurred:", e)
            break

    cv2.destroyAllWindows()

print("Please press (s) to start or (a) to exit.")
while True:
    if keyboard.is_pressed("s"):
        main_app()
        break
