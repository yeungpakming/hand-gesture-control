import cv2
import hand_gesture_recognizer as hgr
import mouse_controller as mc
import time


def main():
    camera = 0
    window_name = "Gesture Control"
    source = cv2.VideoCapture(camera)
    cv2.namedWindow(window_name)
    recognizer = hgr.hand_gesture_recognizer()
    label = hgr.landmark_label
    mouse = mc.mouse_controller()

    while cv2.waitKey(1) != 27:
        has_frame, frame = source.read()
        if not has_frame:
            print("Ignoring empty camera frame.")
            continue
        frame = recognizer.hand_detector(frame)
        mouse.move(recognizer.get_position(label.MIDDLE_FINGER_MCP))
        mouse.left_click(
            recognizer.get_distance(label.INDEX_FINGER_TIP, label.THUMB_TIP)
        )
        cv2.imshow(window_name, cv2.flip(frame, 1))

    source.release()
    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
