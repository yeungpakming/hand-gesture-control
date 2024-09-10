import pyautogui
import math


class mouse_controller:
    def __init__(self, smoothing_factor=0.3, actuation_distance=40, reset_distance=50):
        self.smoothing_factor = smoothing_factor
        self.screen_size = pyautogui.size()
        self.scaling_factor = math.sqrt(self.screen_size[0] * self.screen_size[1])
        self.x = 0.5 * self.screen_size[0]
        self.y = 0.5 * self.screen_size[1]
        self.actuation_distance = actuation_distance
        self.reset_distance = reset_distance
        self.left_click_status = "released"
        self.right_click_status = "released"

    def screen_position(self, position):
        x, y = position
        return ((1 - x) * self.screen_size[0], y * self.screen_size[1])

    def screen_distance(self, distance):
        return distance * self.scaling_factor

    def move(self, position):
        if position == None:
            return
        x_new, y_new = self.screen_position(position)
        x_new = self.smoothing_factor * x_new + (1 - self.smoothing_factor) * self.x
        y_new = self.smoothing_factor * y_new + (1 - self.smoothing_factor) * self.y
        self.x, self.y = x_new, y_new
        pyautogui.moveTo(self.x, self.y, _pause=False)
        return

    def left_click(self, distance):
        if distance == None:
            return
        distance = self.screen_distance(distance)
        if (distance < self.actuation_distance) and (
            self.left_click_status == "released"
        ):
            self.left_click_status = "pressed"
            pyautogui.leftClick(_pause=False)
        if distance > self.reset_distance:
            self.left_click_status = "released"
        return

    def right_click(self, distance):
        if distance == None:
            return
        distance = self.screen_distance(distance)
        if (distance < self.actuation_distance) and (
            self.right_click_status == "released"
        ):
            self.right_click_status = "pressed"
            pyautogui.rightClick(_pause=False)
        if distance > self.reset_distance:
            self.right_click_status = "released"
        return
