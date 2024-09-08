import pyautogui

class mouse_controller:

    def move(self, position):
        x, y = position
        pyautogui.moveTo(x, y, _pause=False)

    def press(self):
        pyautogui.mouseDown()

    def release(self):
        pyautogui.mouseUp()
