import pyautogui

# Function to perform a click at given coordinates and then move to the target location
def click_and_move(x, y, target_x, target_y):
    pyautogui.click(x, y)
    pyautogui.click(target_x, target_y)

# List of coordinates for clicks
click_coordinates = [
    (133, 172),
    (302, 160),
    (455, 178),
    (634, 187),
    (767, 192),
    (949, 186),
    (1098, 180),
    (1127, 175),
    (1312, 183),
    (1465, 183)
]

# Target location
target_x, target_y = 1784, 522

# Clicking loop with target location
for coords in click_coordinates:
    click_and_move(coords[0], coords[1], target_x, target_y)
    pyautogui.PAUSE = 0.2  # Adjust this value to change the time between clicks
