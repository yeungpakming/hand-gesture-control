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

    # the time module is used for checking fps(i commented), i just copy this part of the code from somewhere :P so will remove upon finishing the project; or i'll write my own
    # the text is flipped, because cv2.imshow flipping the frame, still readable so lazy to fix
    # cTime = 0
    # pTime = 0

    while cv2.waitKey(1) != 27:
        has_frame, frame = source.read()
        if not has_frame:
            print("Ignoring empty camera frame.")
            continue
        frame = recognizer.hand_detector(frame)
        mouse.move(recognizer.screen_position(label.MIDDLE_FINGER_MCP))


        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
        # cv2.putText(
        #     frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        # )
        
        cv2.imshow(window_name, cv2.flip(frame, 1))

    source.release()
    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
