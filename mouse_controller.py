import pyautogui
import math


class mouse_controller:
    def __init__(self, smoothing_factor=0.3, actuation_distance=40, reset_distance=60):
        self.smoothing_factor = smoothing_factor
        self.screen_size = pyautogui.size()
        self.scaling_factor = math.sqrt(self.screen_size[0] * self.screen_size[1])
        self.x = 0.5 * self.screen_size[0]
        self.y = 0.5 * self.screen_size[1]
        self.actuation_distance = actuation_distance
        self.reset_distance = reset_distance
        self.click_status = "released"

    def screen_position(self, position):
        x, y = position
        return ((1 - x) * self.screen_size[0], y * self.screen_size[1])

    def screen_distance(self, distance):
        return distance * self.scaling_factor

    def move(self, position):
        new_x, new_y = self.screen_position(position)
        new_x = self.smoothing_factor * new_x + (1 - self.smoothing_factor) * self.x
        new_y = self.smoothing_factor * new_y + (1 - self.smoothing_factor) * self.y
        self.x, self.y = new_x, new_y
        pyautogui.moveTo(self.x, self.y, _pause=False)

    def left_click(self, distance):
        distance = self.screen_distance(distance)
        if distance < self.actuation_distance:
            self.click_status = "pressed"
        if self.click_status == "pressed":
            pyautogui.mouseDown(_pause=False)
        if (distance > self.reset_distance) and (self.click_status == "pressed"):
            self.click_status = "released"
            pyautogui.mouseUp(_pause=False)
