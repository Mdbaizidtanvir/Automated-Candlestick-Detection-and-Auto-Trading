from ultralytics import YOLO
import cv2
import numpy as np
from mss import mss
import keyboard
import time
import pyautogui
import os

# Initialize YOLO model with pre-trained weights
model = YOLO(r'E:\train8\train8\weights\best.pt')

# Define screen dimensions for capture
screen = {"top": 0, "left": 0, "width": 1920, "height": 1080}
sct = mss()

# Create a resizable window
cv2.namedWindow('Real-Time Object Recognition', cv2.WINDOW_NORMAL)

cooldown = 60  # Cooldown period in seconds
last_click_time = 0

# Function to click with shift key and 'd'
def click_with_shift_d(x, y):
    pyautogui.click(x, y)  # Click at the detected coordinates (higher or lower)
    time.sleep(0.2)  # 5-second cooldown
    pyautogui.click(1803, 328)  # Click at coordinates (1803, 328)
    time.sleep(0.5)  # 5-second cooldown
    pyautogui.click(1317, 529)  # Click at coordinates (1317, 529)
    time.sleep(0.7)  # 5-second cooldown
    pyautogui.click(1770, 839)  # Click at coordinates (1770, 839)

# Function to press Shift + A
def press_shift_a():
    pyautogui.keyDown('shift')
    pyautogui.press('a')
    pyautogui.keyUp('shift')

# Path to template directories
template_directory_higher = "./bi-images/higher"
template_directory_lower = "./bi-images/lower"
template_directory_skip = "./bi-images/skip"  # Corrected path separator

# Read template images from the directories
template_images_higher = [cv2.imread(os.path.join(template_directory_higher, filename)) for filename in os.listdir(template_directory_higher) if filename.endswith(".png")]
template_images_lower = [cv2.imread(os.path.join(template_directory_lower, filename)) for filename in os.listdir(template_directory_lower) if filename.endswith(".png")]
template_images_skip = [cv2.imread(os.path.join(template_directory_skip, filename)) for filename in os.listdir(template_directory_skip) if filename.endswith(".png")]

def main_app():
    global cooldown, last_click_time
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
                        click_with_shift_d(click_x, click_y)
                        last_click_time = time.time()

                elif object_name == "sell":
                    print("sell is found")
                    if time.time() - last_click_time > cooldown:
                        click_x = 1757
                        click_y = 572
                        press_shift_a()
                        last_click_time = time.time()

                elif object_name == "skip":
                    print("skip is found")
                    # Perform shift + A action here
                    press_shift_a()

        # Display processed frame in a resizable window
        cv2.imshow('Real-Time Object Recognition', results[0].plot())

        if cv2.waitKey(25) & 0xFF == ord('q'):  # Exit on 'q'
            break

        if keyboard.is_pressed("a"):
            print("Exiting...")
            break

print("Please press (s) to start or (a) to exit.")
while True:
    if keyboard.is_pressed("s"):
        main_app()

cv2.destroyAllWindows()
