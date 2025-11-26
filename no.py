import os
import cv2
import numpy as np
import pyautogui

# Template directories
template_directory_higher = "./bi-images/higher/"
template_directory_lower = "./bi-images/lower/"
template_directory_skip = "./bi-images/skip/"

# Read template images from the directories
template_images_higher = [cv2.imread(os.path.join(template_directory_higher, filename)) for filename in os.listdir(template_directory_higher) if filename.endswith(".png")]
template_images_lower = [cv2.imread(os.path.join(template_directory_lower, filename)) for filename in os.listdir(template_directory_lower) if filename.endswith(".png")]
template_images_skip = [cv2.imread(os.path.join(template_directory_skip, filename)) for filename in os.listdir(template_directory_skip) if filename.endswith(".png")]

# Template matching method
template_matching_method = cv2.TM_CCOEFF_NORMED

# Click coordinates for higher and lower
click_coordinates = {
    "higher": (1781, 550),
    "lower": (1757, 572)
}

def template_matched(templates, frame_bgr):
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    for template_image in templates:
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(frame_gray, template_gray, template_matching_method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.8:
            # Return not only the match status but also the match location
            return True, max_loc, template_image.shape[1], template_image.shape[0]
    return False, None, None, None

def click_with_shift_d(x, y):
    pyautogui.click(x, y)
    for _ in range(2):  # Execute twice
        pyautogui.keyDown('shift')
        pyautogui.press('d')  # Press D key
        pyautogui.keyUp('shift')

def press_shift_a():
    pyautogui.keyDown('shift')
    for _ in range(30):
        pyautogui.press('a')
    pyautogui.keyUp('shift')

def main_app():
    # Load image
    frame_bgr = cv2.imread("bi-images/sample_image.png")

    if frame_bgr is None:
        print("Error: Failed to load the sample image.")
        return

    match_skip, lock, wk, hk = template_matched(template_images_skip, frame_bgr)
    if match_skip:
        top_left = lock
        bottom_right = (top_left[0] + wk, top_left[1] + hk)
        cv2.rectangle(frame_bgr, top_left, bottom_right, (0, 255, 0), 2)
        print("Skip image detected and bordered.")
        press_shift_a()  # Press Shift + A 30 times
    else:
        print("Skip image is not bordered. Clicks will now work.")

        match, loc, w, h = template_matched(template_images_higher, frame_bgr)
        if match:
            top_left = loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame_bgr, top_left, bottom_right, (0, 255, 0), 2)
            print("Higher image detected and bordered.")
            click_with_shift_d(click_coordinates["higher"][0], click_coordinates["higher"][1])

        match1, loc1, w1, h1 = template_matched(template_images_lower, frame_bgr)
        if match1:
            top_left = loc1
            bottom_right = (top_left[0] + w1, top_left[1] + h1)
            cv2.rectangle(frame_bgr, top_left, bottom_right, (0, 0, 255), 2)
            print("Lower image detected and bordered.")
            click_with_shift_d(click_coordinates["lower"][0], click_coordinates["lower"][1])

    cv2.imshow('Object Detection', frame_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_app()
