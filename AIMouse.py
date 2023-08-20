# Imports
import mediapipe as mp
import cv2
import time
import math
import pyautogui as pyg
import winsound
import numpy as np
import sys
import screen_brightness_control as sbc

width, heigth = pyg.size()

"""This Program is used to control the functions of mouse and keyboard using Media Pipe and OpenCV.

--------------------###############################################-------------------------
The class Hand Detector assigns the right and left hands and calculates their 22 landmarks"""


class handDetector:
    # Initialising the hands module of mediapipe
    def __init__(
        self, mode=False, max_hand=2, detect_con=0.5, track_con=0.5, model_complexity=1
    ):
        self.mode = mode
        self.max_hand = max_hand
        self.detect_con = detect_con
        self.track_con = track_con
        self.model_complexity = model_complexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            self.max_hand,
            self.model_complexity,
            self.detect_con,
            self.track_con,
        )
        self.mpdraw = mp.solutions.drawing_utils

    """Draws hand but needs to convert BRG image of OpenCV to RGB to be used by mediapipe"""

    def draw_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        if self.result.multi_hand_landmarks:
            for i, handlm in enumerate(self.result.multi_hand_landmarks):
                if draw:
                    self.mpdraw.draw_landmarks(
                        img, handlm, self.mpHands.HAND_CONNECTIONS
                    )

        return img

    """findPosition() caclulates the values of all the landmarks in hands and assigns them to the respective hand in 
    pixel coordinate system, making the data much cleaner"""

    def findPosition(self, img, draw=True):
        lmdict = {}
        h, w, c = img.shape
        if self.result.multi_hand_landmarks:
            for handType, handLms in zip(
                self.result.multi_handedness, self.result.multi_hand_landmarks
            ):
                ## lmList
                mylmList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([id, px, py])
                if handType.classification[0].label == "Right":
                    lmdict["Left"] = mylmList
                else:
                    lmdict["Right"] = mylmList
        return lmdict

    """FindFingers() is used to detect which fingers in both hands are raised and append them in a list allowing for easier and efficient detecting"""

    def FindFingers(self, lmdict: dict):
        indexids = [4, 8, 12, 16, 20]
        fingers_l = []
        fingers_r = []
        self.all_fingers = []

        if len(lmdict) != 0:
            if "Left" in lmdict:
                fingers_l = [0, 0, 0, 0, 0]
                midlandlt = (lmdict["Left"][indexids[0]][1] + lmdict["Left"][5][1]) / 2
                if lmdict["Left"][indexids[0]][1] < midlandlt:
                    fingers_l[0] = 1

                for id in range(1, 5):
                    if (
                        lmdict["Left"][indexids[id]][2]
                        < lmdict["Left"][indexids[id] - 2][2]
                    ):
                        fingers_l[id] = 1

            if "Right" in lmdict:
                fingers_r = [0, 0, 0, 0, 0]
                midlandrt = (
                    lmdict["Right"][indexids[0]][1] + lmdict["Right"][5][1]
                ) / 2
                if lmdict["Right"][indexids[0]][1] > midlandrt:
                    fingers_r[0] = 1

                for id in range(1, 5):
                    if (
                        lmdict["Right"][indexids[id]][2]
                        < lmdict["Right"][indexids[id] - 2][2]
                    ):
                        fingers_r[id] = 1

        self.all_fingers.append(fingers_l)
        self.all_fingers.append(fingers_r)
        return self.all_fingers

    """cl() and cr() calculate the additional landmark for both left and right hands respectively"""

    def cl(self, lmdict, lm1=9, lm2=0):
        x1, y1 = lmdict["Left"][lm1][1], lmdict["Left"][lm1][2]
        x2, y2 = lmdict["Left"][lm2][1], lmdict["Left"][lm2][2]
        cl = ((x1 + x2) // 2, (y1 + y2) // 2)
        return cl

    def cr(self, lmdict, lm1=9, lm2=0):
        x3, y3 = lmdict["Right"][lm1][1], lmdict["Right"][lm1][2]
        x4, y4 = lmdict["Right"][lm2][1], lmdict["Right"][lm2][2]
        cr = ((x3 + x4) // 2, (y3 + y4) // 2)
        return cr

    """Function FindDist() calculates the distance between Mark1 and Mark2 (landmarks) of the hand """

    def FindDist(self, Mark1, Mark2, dictlm: dict):
        if "Left" in dictlm:
            x1, y1 = dictlm["Left"][Mark1][1:]
            x2, y2 = dictlm["Left"][Mark2][1:]
        elif "Right" in dictlm:
            x1, y1 = dictlm["Right"][Mark1][1:]
            x2, y2 = dictlm["Right"][Mark2][1:]

        return math.hypot(x2 - x1, y2 - y1)


"""Gesture Controller Class Maps the gestures to Their Functionality"""


class GestureControl(handDetector):

    """Class Attributes also act as initialising variables for the gestures"""

    pcr = (
        0,
        0,
    )  # pcr = previous values cr is used to refer to right hands additional landmark
    pcl = (
        0,
        0,
    )  # pcl = previous values cl is used to refer to left hands additional landmark
    CB = sbc.get_brightness()
    C_B = CB[0]

    """MousePointer interploates the distance moved by the finger in a box and maps it to the screen according
    The Interplotation makes it much more smoother and easier to use"""

    def MousePointer(self, lmdict, img, mark=8):
        framR = 50
        wcam = 640
        hcam = 480

        cv2.rectangle(
            img, (framR, framR), (wcam - framR, hcam - framR), (255, 0, 255), 2
        )
        if "Left" in lmdict:
            x1, y1 = lmdict["Left"][mark][1:]
        if "Right" in lmdict:
            x1, y1 = lmdict["Right"][mark][1:]

        x3 = np.interp(x1, (framR, wcam - framR), (0, width))
        y3 = np.interp(y1, (framR, hcam - framR), (0, heigth))
        pyg.moveTo((width - x3), y3, 0.12)

    """Volume() calculates the distance moved in Y axis and changes volume by 10 units accordingly"""

    def Volume(self, lmdict):
        ccl = self.cl(lmdict)
        if (ccl[1] - self.pcl[1]) < 0 and self.pcl != (0, 0):
            pyg.press(["volumeup", "volumeup", "volumeup", "volumeup", "volumeup"])
        elif (ccl[1] - self.pcl[1]) > 0 and self.pcl != (0, 0):
            pyg.press(
                ["volumedown", "volumedown", "volumedown", "volumedown", "volumedown"]
            )
        cv2.waitKey(100)
        self.pcl = ccl

    """Brightness_Ctrl() calculates the distance moved in Y axis and changes brightness by 5 units accordingly"""

    def Brightness_Ctrl(self, lmdict):
        ccr = self.cr(lmdict)

        if (ccr[1] - self.pcr[1]) < 0 and self.pcr != (0, 0):
            self.C_B += 5
            if self.C_B >= 100:
                self.C_B = 100

        elif (ccr[1] - self.pcr[1]) > 0 and self.pcr != (0, 0):
            self.C_B -= 5
            if self.C_B <= 0:
                self.C_B = 0

        sbc.set_brightness(self.C_B)
        cv2.waitKey(100)
        self.pcr = ccr

    """Vscroll() calculates the distance moved in Y axis and scrolls the screen vertically"""

    def VScroll(self, lmdict):
        ccl = self.cl(lmdict)
        if (ccl[1] - self.pcl[1]) < 0 and self.pcl != (0, 0):
            pyg.scroll(-120)
        elif (ccl[1] - self.pcl[1]) > 0 and self.pcl != (0, 0):
            pyg.scroll(120)
        cv2.waitKey(100)
        self.pcl = ccl

    """Hscroll() calculates the distance moved in X axis and scrolls the screen vertically"""

    def HScroll(self, lmdict):
        ccr = self.cr(lmdict)
        if (ccr[0] - self.pcr[0]) < 0 and self.pcr != (0, 0):
            pyg.keyDown("shift")
            pyg.scroll(120)
            cv2.waitKey(50)
            pyg.keyUp("shift")

        elif (ccr[0] - self.pcr[0]) > 0 and self.pcr != (0, 0):
            pyg.keyDown("shift")
            pyg.scroll(-120)
            cv2.waitKey(50)
            pyg.keyUp("shift")

        cv2.waitKey(50)
        self.pcr = ccr

    """This function maps all gestures to the respective functions
   -----------------------------############################################------------------------------------"""

    def Gestures(self, fingerlist, lmdict, img):
        # For Opening Desktop
        if fingerlist == [[0, 1, 1, 0, 1], [0, 1, 1, 0, 1]]:
            pyg.keyUp("alt")
            pyg.hotkey("win", "d")
            cv2.waitKey(2000)
            return

        # For opening file explorer
        if fingerlist == [[0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]:
            pyg.keyUp("alt")
            pyg.hotkey("win", "e")
            cv2.waitKey(1500)
            return

        # For New Folder Creation
        if fingerlist == [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]] or fingerlist == [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ]:
            pyg.hotkey("ctrl", "shift", "n")
            cv2.waitKey(1000)
            return

        # For Shutdown
        if fingerlist == [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]] or fingerlist == [
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
        ]:
            cv2.destroyAllWindows()
            sys.exit()

        # For Volume Control With Left Hand
        if len(fingerlist[1]) == 0 and fingerlist[0] == [0, 1, 1, 1, 1]:
            self.Volume(lmdict)
            return

        # For Brightness Control With Right Hand
        if len(fingerlist[0]) == 0 and fingerlist[1] == [0, 1, 1, 1, 1]:
            self.Brightness_Ctrl(lmdict)
            return

        # For Vertical Scroll With Left Hand.
        if (
            len(fingerlist[1]) == 0
            and (fingerlist[0] == [0, 1, 1, 0, 0] or fingerlist[0] == [1, 1, 1, 0, 0])
            and (self.FindDist(8, 12, lmdict) < 40)
        ):
            self.VScroll(lmdict)
            return

        # For Horizontal With Right Hand
        if (
            len(fingerlist[0]) == 0
            and (fingerlist[1] == [0, 1, 1, 0, 0] or fingerlist[1] == [1, 1, 1, 0, 0])
            and (self.FindDist(8, 12, lmdict) < 40)
        ):
            self.HScroll(lmdict)
            return

        """-----------Conditions for Both Hands------------"""
        for i in fingerlist:
            if len(i) != 0:
                # Grabbing or Drag/Drop Functionality
                if i == [0, 0, 0, 0, 0] or i == [1, 0, 0, 0, 0]:
                    pyg.mouseDown()
                    self.MousePointer(lmdict, img, mark=5)

                # For Taking ScreenShots (Can be found default pictures Directory with ScreenShots Folder)
                if i == [1, 0, 0, 0, 1]:
                    pyg.keyUp("alt")
                    pyg.hotkey("win", "prntscrn")
                    winsound.PlaySound("*", winsound.SND_ALIAS)
                    cv2.waitKey(1500)

                # For moving the mouse
                if i == [0, 1, 0, 0, 0] or i == [1, 1, 0, 0, 0]:
                    pyg.keyUp("win")
                    self.MousePointer(lmdict, img)

                # For Left Clicking
                if (i == [1, 1, 1, 0, 0] or i == [0, 1, 1, 0, 0]) and (
                    self.FindDist(8, 12, lmdict) > 50
                ):
                    pyg.click(clicks=2)
                    pyg.keyUp("alt")
                    cv2.waitKey(1000)

                # For Right clicking
                if i == [1, 1, 1, 1, 0] or i == [0, 1, 1, 1, 0]:
                    pyg.click(button="right")
                    cv2.waitKey(1000)

                # For Changings Tabs
                if i == [0, 0, 1, 1, 1] or i == [1, 0, 1, 1, 1]:
                    pyg.keyUp("win")
                    pyg.keyDown("alt")
                    pyg.press("Tab")
                    cv2.waitKey(100)

                # Using Window Button
                if i == [0, 0, 1, 1, 0] or i == [1, 0, 1, 1, 0]:
                    pyg.press("win")
                    cv2.waitKey(1000)

                # For Maximizing the Window
                if i == [1, 1, 0, 0, 1]:
                    pyg.keyDown("win")
                    pyg.hotkey("up")
                    cv2.waitKey(500)

                # For Minimizing the Window
                if i == [0, 1, 0, 0, 1]:
                    pyg.hotkey("win", "down")
                    cv2.waitKey(500)

                # Neutral State
                if i == [1, 1, 1, 1, 1]:
                    pcr = (0, 0)
                    pcl = (0, 0)
                    pyg.keyUp("alt")
                    pyg.mouseUp()
                    pyg.keyUp("win")

            """The added If clauses in the Gestures like if i==[0,0,1,1,0] or i==[1,0,1,1,0]: for line 287
            This is added to compensate for miscalculations during hand detection by mediapipe module."""
            """-----------------###############################-----------------------------"""


"""Main Program"""


def main():
    pyg.FAILSAFE = False
    cap = cv2.VideoCapture(0)
    wcam = cap.get(3)
    hcam = cap.get(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wcam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hcam)
    detector = handDetector()
    controller = GestureControl()
    ptime = 0
    ctime = 0
    while True:
        success, img = cap.read()

        img = detector.draw_hands(img)
        lmdict = detector.findPosition(img)

        if len(lmdict) != 0:
            if "Left" in lmdict:
                cl = detector.cl(lmdict)
                cv2.circle(img, (cl[0], cl[1]), 5, (0, 0, 225), cv2.FILLED)
            if "Right" in lmdict:
                cr = detector.cr(lmdict)
                cv2.circle(img, (cr[0], cr[1]), 5, (0, 0, 225), cv2.FILLED)

            cv2.waitKey(10)
            Fingers = detector.FindFingers(lmdict)
            controller.Gestures(Fingers, lmdict, img)

        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        img = cv2.flip(img, 1)
        cv2.putText(
            img,
            "FPS: " + str(int(fps)),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )

        cv2.imshow("Virtual Mouse", img)


if __name__ == "__main__":
    main()
