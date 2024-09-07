import cv2
import mediapipe as mp
from enum import IntEnum
import math
import pyautogui
import time

camera = 0
window_name = "Gesture Control"


class landmark_label(IntEnum):
    WRIST = 0
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12


class hand_gesture_recognizer:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hand_landmarker = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.screen_size = pyautogui.size()

    def hand_detector(self, frame):
        self.width, self.height, self.channel = frame.shape
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hand_landmarker.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.result.multi_hand_landmarks:
            self.has_finger = True
            for hand_landmark in self.result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmark,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame

    def get_position(self, label):
        if self.result.multi_hand_landmarks:
            for hand_landmark in self.result.multi_hand_landmarks:
                return (
                    hand_landmark.landmark[label].x,
                    hand_landmark.landmark[label].y,
                )
        return (None, None)

    def distance(self, label_1, label_2):
        x_1, y_1 = self.get_position(label_1)
        x_2, y_2 = self.get_position(label_2)
        return math.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

    def convert_position(self, position):
        x, y = position
        if x is None:
            return (x, y)
        return ((1 - x) * self.screen_size[0], y * self.screen_size[1])

    def mouse_move(self, label):
        x, y = self.convert_position(self.get_position(label))
        pyautogui.moveTo(x, y, _pause=False)


def main():
    source = cv2.VideoCapture(camera)
    cv2.namedWindow(window_name)
    recognizer = hand_gesture_recognizer()
    cTime = 0
    pTime = 0

    while cv2.waitKey(1) != 27:
        has_frame, frame = source.read()
        if not has_frame:
            print("Ignoring empty camera frame.")
            continue
        frame = recognizer.hand_detector(frame)
        recognizer.mouse_move(landmark_label.WRIST)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )

        cv2.imshow(window_name, cv2.flip(frame, 1))

    source.release()
    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()


#add hand.status to deal with position being None