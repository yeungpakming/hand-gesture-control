import pyautogui
import math


class mouse_controller:
    def __init__(self, smoothing_factor=0.3, actuation_distance=50):
        self.smoothing_factor = smoothing_factor
        self.screen_size = pyautogui.size()
        self.scaling_factor = math.sqrt(self.screen_size[0] * self.screen_size[1])
        self.x = 0.5 * self.screen_size[0]
        self.y = 0.5 * self.screen_size[1]
        self.actuation_distance = actuation_distance
        self.distance = 100
        self.click_status = True

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
        if self.distance > self.actuation_distance:
            self.status = True
        self.distance = self.screen_distance(distance)
        if self.distance < self.actuation_distance and self.status:
            pyautogui.click(_pause=False)
            self.status = False
