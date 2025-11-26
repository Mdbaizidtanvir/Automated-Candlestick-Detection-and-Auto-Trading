import os  # For interacting with the operating system
import cv2  # For computer vision tasks
import numpy as np  # For numerical operations
from mss import mss  # For screen capturing
import keyboard  # For keyboard inputs
import time  # For time-related operations
import pyautogui  # For simulating mouse and keyboard inputs
from ultralytics import YOLO  # For object detection using YOLO

# Initialize screen dimensions for capture
screen = {"top": 0, "left": 0, "width": 1920, "height": 1080}
sct = mss()

# Template directories
template_directory_buy = "./bi-images/higher/"
template_directory_sell = "./bi-images/lower/"
template_directory_skip = "./bi-images/skip/"

# Read template images from the directories
template_images_buy = [cv2.imread(os.path.join(template_directory_buy, filename), cv2.IMREAD_GRAYSCALE) for filename in os.listdir(template_directory_buy) if filename.endswith(".png")]
template_images_sell = [cv2.imread(os.path.join(template_directory_sell, filename), cv2.IMREAD_GRAYSCALE) for filename in os.listdir(template_directory_sell) if filename.endswith(".png")]
template_images_skip = [cv2.imread(os.path.join(template_directory_skip, filename), cv2.IMREAD_GRAYSCALE) for filename in os.listdir(template_directory_skip) if filename.endswith(".png")]

# Template matching method
template_matching_method = cv2.TM_CCOEFF_NORMED

# Cooldown periods
click_cooldown_buy = 65
click_cooldown_sell =  120
last_click_time_buy = 0
last_click_time_sell = 0
last_shift_tab_time = 0

# Click coordinates for buy and sell
click_coordinates = {
    "buy": (1781, 550),
    "sell": (1757, 572),
    "hourly": [(91, 59), (1889, 378)]  # Hourly click coordinates
}

# YOLO model initialization
model = YOLO(r'E:\train8\train8\weights\best.pt')

# Function to check if template matches
def template_matched(templates, frame_bgr, is_skip=False):
    if is_skip:
        for template_image in templates:
            res = cv2.matchTemplate(frame_bgr, template_image, template_matching_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.77:
                # Return not only the match status but also the match location
                return True, max_loc, template_image.shape[1], template_image.shape[0]
        return False, None, None, None
    else:
        for template_image in templates:
            res = cv2.matchTemplate(frame_bgr, template_image, template_matching_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.8:
                # Return not only the match status but also the match location
                return True, max_loc, template_image.shape[1], template_image.shape[0]
        return False, None, None, None

# Function to click with shift key and 'd'
def click_with_shift_d(x, y):
    pyautogui.click(x, y)  # Click at the detected coordinates (buy or sell)
    time.sleep(0.1)  # 5-second cooldown
    pyautogui.click(1803, 328)  # Click at coordinates (1803, 328)
    time.sleep(0.3)  # 5-second cooldown
    pyautogui.click(1317, 529)  # Click at coordinates (1317, 529)
    time.sleep(0.5)  # 5-second cooldown
    pyautogui.click(1824, 838)  # Click at coordinates (1824, 838)

def press_shift_a():
    pyautogui.keyDown('shift')  # Press and hold the Shift key
    for _ in range(10):  # Press 'a' 30 times
        pyautogui.press('a')
    pyautogui.keyUp('shift')  # Release the Shift key

# Main application function
def main_app():
    global last_click_time_buy, last_click_time_sell, last_shift_tab_time

    while True:
        frame_img = np.array(sct.grab(screen))
        frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_BGRA2BGR)

        match_skip, lock, wk, hk = template_matched(template_images_skip, frame_bgr, is_skip=True)
        if match_skip:
            top_left = lock
            bottom_right = (top_left[0] + wk, top_left[1] + hk)
            cv2.rectangle(frame_img, top_left, bottom_right, (0, 255, 0), 2)
            print("Skip image detected and bordered.")
            press_shift_a()  # Press Shift + A 30 times

            # Check if it's been 10 minutes since the skip image was detected
            current_time = time.time()
            if current_time - last_shift_tab_time >= 600:
                pyautogui.hotkey('shift', 'tab')  # Press Shift + Tab once
                last_shift_tab_time = current_time  # Update the last Shift + Tab time

        else:
            print("Skip image is not bordered. Clicks will now work.")

            match, loc, w, h = template_matched(template_images_buy, frame_bgr)
            if match:
                top_left = loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(frame_img, top_left, bottom_right, (0, 255, 0), 2)
                print("Buy image detected and bordered.")
                current_time = time.time()
                if current_time - last_click_time_buy > click_cooldown_buy:
                    click_with_shift_d(click_coordinates["buy"][0], click_coordinates["buy"][1])
                    last_click_time_buy = current_time
            else:
                print("Not Buy Image")

            match1, loc1, w1, h1 = template_matched(template_images_sell, frame_bgr)
            if match1:
                top_left = loc1
                bottom_right = (top_left[0] + w1, top_left[1] + h1)
                cv2.rectangle(frame_img, top_left, bottom_right, (0, 0, 255), 2)
                print("Sell image detected and bordered.")
                current_time = time.time()
                if current_time - last_click_time_sell > click_cooldown_sell:
                    click_with_shift_d(click_coordinates["sell"][0], click_coordinates["sell"][1])
                    last_click_time_sell = current_time
            else:
                print("Not Sell Image")

        cv2.imshow('Real-Time Object Recognition', frame_img)

        if cv2.waitKey(25) & 0xFF == ord('q'):  # Exit on 'q'
            break

        if keyboard.is_pressed("a"):
            print("Exiting...")
            break

        if keyboard.is_pressed("s"): # For exit
            print("Exiting...")
            break


# Main function to handle the timer for hourly clicks
def start_hourly_clicks():
    while True:
        # Perform 3 clicks at the first position
        for _ in range(1):
            pyautogui.click(click_coordinates["hourly"][0][0], click_coordinates["hourly"][0][1])
            time.sleep(1)  # 1-second cooldown between clicks

        # Wait for 5 seconds
        time.sleep(5)

        # Perform 2 clicks at the second position
        for _ in range(2):
            pyautogui.click(click_coordinates["hourly"][1][0], click_coordinates["hourly"][1][1])
            time.sleep(1)  # 1-second cooldown between clicks

        # Perform the main app function
        main_app()
        # Sleep for an hour
        time.sleep(3600)  # 3600 seconds = 1 hour

print("Please press (s) to start or (a) to exit.")
while True:
    if keyboard.is_pressed("s"):
        start_hourly_clicks()

cv2.destroyAllWindows()
