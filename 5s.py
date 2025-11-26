from ultralytics import YOLO
import cv2
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

# Create a resizable window
cv2.namedWindow('Real-Time Object Recognition', cv2.WINDOW_NORMAL)

cooldown = 5  # Cooldown period in seconds
last_click_time = 0

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
                        click_x = 1774
                        click_y = 459
                        pyautogui.click(click_x, click_y)
                        last_click_time = time.time()
                        # Pressing Shift + D
                        pyautogui.keyDown('shift')
                        pyautogui.press('d')
                        pyautogui.keyUp('shift')

                if object_name == "sell":
                    print("sell is found")
                    if time.time() - last_click_time > cooldown:
                        click_x = 1775
                        click_y = 529
                        pyautogui.click(click_x, click_y)
                        last_click_time = time.time()
                        # Pressing Shift + D
                        pyautogui.keyDown('shift')
                        pyautogui.press('d')
                        pyautogui.keyUp('shift')

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
