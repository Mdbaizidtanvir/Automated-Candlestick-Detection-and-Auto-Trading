import os
import cv2
from ultralytics import YOLO
import numpy as np
from mss import mss
import keyboard
import time
import pyautogui

# Initialize YOLO model with pre-trained weights
model = YOLO(r'E:\train8\train8\weights\best.pt')

# Define screen dimensions for capture
screen = {"top": 0, "left": 0, "width": 1920, "height": 1080}
sct = mss()

# Directory containing template images for skip
template_directory_skip = "E:/advanced detection/bi-images/skip/"
# Read template images from the directory
template_images_skip = [cv2.imread(os.path.join(template_directory_skip, filename)) for filename in os.listdir(template_directory_skip) if filename.endswith(".png")]
# Define the template matching method
template_matching_method = cv2.TM_CCOEFF_NORMED

# Create a resizable window
cv2.namedWindow('Real-Time Object Recognition', cv2.WINDOW_NORMAL)

cooldown = 60  # Cooldown period in seconds
last_click_time = 0
last_shuffle_time = time.time()

def main_app():
    global cooldown, last_click_time, last_shuffle_time

    while True:
        # Screen capture and conversion for OpenCV processing
        sct_img = sct.grab(screen)
        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

        # Object detection and tracking
        results = model.predict(frame, conf=0.15)
        for result in results:
            if result.boxes:
                box = result.boxes[0]
                class_id = int(box.cls)
                object_name = model.names[class_id]
                if object_name == "buy":
                    print("buy is found")
                    if time.time() - last_click_time > cooldown:
                        click_x = 1781
                        click_y = 550
                        pyautogui.click(click_x, click_y)
                        last_click_time = time.time()
                        # Pressing Shift + D
                        pyautogui.keyDown('shift')
                        pyautogui.press('d')
                        pyautogui.keyUp('shift')

                if object_name == "sell":
                    print("sell is found")
                    if time.time() - last_click_time > cooldown:
                        click_x = 1757
                        click_y = 572
                        pyautogui.click(click_x, click_y)
                        last_click_time = time.time()
                        # Pressing Shift + D
                        pyautogui.keyDown('shift')
                        pyautogui.press('d')
                        pyautogui.keyUp('shift')

        # Template matching for skip
        for template_image in template_images_skip:
            # Perform template matching
            result = cv2.matchTemplate(frame, template_image, template_matching_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Check if a match is found
            if max_val > 0.8:  # Adjust threshold as needed
                print("Skip detected!")
                # Pressing Shift + A multiple times with a slight delay
                for _ in range(10):
                    pyautogui.keyDown('shift')
                    pyautogui.press('a')
                    pyautogui.keyUp('shift')
                    time.sleep(0.05)  # Adjust the delay as needed

        # Check if 10 minutes have passed for shuffling
        if time.time() - last_shuffle_time > 600:
            print("Shuffling...")
            pyautogui.hotkey('shift', 'tab')
            last_shuffle_time = time.time()

        # Display processed frame in a resizable window
        cv2.imshow('Real-Time Object Recognition', results[0].plot())

        if cv2.waitKey(25) & 0xFF == ord('q'):  # Exit on 'q'
            break

        if keyboard.is_pressed("a"):
            print("Exiting...")
            break

        if keyboard.is_pressed("s"): # For exit
            print("Exiting...")
            break

print("Please press (s) to start or (a) to exit.")
while True:
    if keyboard.is_pressed("s"):
        main_app()

cv2.destroyAllWindows()
