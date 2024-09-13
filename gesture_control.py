import cv2
import hand_gesture_recognizer as hgr
import mouse_controller as mc
import time


class gesture_control:
    def __init__(
        self,
        camera=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smoothing_factor=0.3,
        padding=0.3,
        actuation_distance=35,
        reset_distance=45,
        reminder=True,
    ):
        self.camera = camera
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smoothing_factor = smoothing_factor
        self.padding = padding
        self.actuation_distance = actuation_distance
        self.reset_distance = reset_distance
        self.reminder = reminder
        self.time_old = 0
        self.time_new = 0

    def fps_display(self, frame):
        self.time_new = time.time()
        fps = 1 / (self.time_new - self.time_old)
        self.time_old = self.time_new
        cv2.putText(
            frame,
            str(int(fps)),
            (10, 30),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (115, 255, 0),
            3,
        )
        return frame

    def run(self):
        message = "please press the escape key to quit the program"
        if self.reminder:
            print(message)
        source = cv2.VideoCapture(self.camera)
        window_name = "Gesture Control"
        cv2.namedWindow(window_name)

        recognizer = hgr.hand_gesture_recognizer(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        label = hgr.landmark_label
        mouse = mc.mouse_controller(
            smoothing_factor=self.smoothing_factor,
            padding=self.padding,
            actuation_distance=self.actuation_distance,
            reset_distance=self.reset_distance,
        )

        while cv2.waitKey(1) != 27:
            has_frame, frame = source.read()
            if not has_frame:
                print("Ignoring empty camera frame.")
                continue
            frame = recognizer.hand_detector(frame)
            mouse.move(recognizer.get_position(label.WRIST))
            mouse.left_click(
                recognizer.get_distance(label.INDEX_FINGER_TIP, label.THUMB_TIP)
            )
            mouse.right_click(
                recognizer.get_distance(label.MIDDLE_FINGER_TIP, label.THUMB_TIP)
            )
            cv2.imshow(window_name, self.fps_display(cv2.flip(frame, 1)))

        source.release()
        cv2.destroyWindow(window_name)


def main():
    gc = gesture_control()
    gc.run()


if __name__ == "__main__":
    main()
