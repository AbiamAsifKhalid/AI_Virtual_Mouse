# Imports
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import AIMouse as ts
import subprocess
from tkinter import ttk

# Initialisations for GUI
win = tk.Tk()
style = ttk.Style()
style.theme_use("clam")
style.configure("C.TButton", font=("Elephant", 12))
style.map(
    "C.TButton",
    foreground=[("!active", "black"), ("pressed", "white"), ("active", "white")],
    background=[("!active", "white"), ("pressed", "black"), ("active", "black")],
)

win.geometry("1080x720+0+0")
color = [(255, 0), (255, 0), (255, 0)]  # For generating colors in gradient

""" This function will create a rectangular gradient on the given canvas"""


def create_gradient(canvas, x1, y1, x2, y2, colors):
    width = x2 - x1
    height = y2 - y1
    for i in range(height):
        r, g, b = [int(c1 + (c2 - c1) * i / height) for c2, c1 in colors]
        color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        canvas.create_line(x1, y1 + i, x2, y1 + i, fill=color, width=1)


# Title of the gui window
win.title("Welcome To Virtual Mouse")
frame_1 = tk.Canvas(win, width=1080, height=720)

create_gradient(frame_1, 0, 0, 1080, 720, color)

# This creates a rectangle on the canvas with the text 'Welcome to "AI" Virtual Mouse in it'
frame_1.create_rectangle(320, 25, 790, 100, width=3, outline="white")
frame_1.create_text(
    550,
    65,
    text='Welcome to "AI" Virtual Mouse',
    font=("Algerian", 22, "italic"),
    fill="white",
)
frame_1.create_text(
    180, 630, text="2021_MC_16 (Muhammad Usman Noor)", font=("Elephant", 12)
)
frame_1.create_text(
    170, 660, text="2021_MC_40 (Abiam Asif Khalid)", font=("Elephant", 12)
)
frame_1.place(x=0, y=0)
frame_1.pack()


# Function kills this program and transfer the Functionality to the Main Program
def main_func():
    cap.release()
    win.destroy()
    cv2.waitKey(1000)
    ts.main()


# Funtion openManual will open a manual for the user to guide them for the AI mouse.
def openManual():
    path = "User Manual.pdf"
    subprocess.Popen([path], shell=True)


# First Button Allows us to go into the Main Program
b_1 = ttk.Button(win, text="Continue", style="C.TButton", command=main_func)
b_1.place(x=875, y=585)

# Second Button to Allow the opening of User Manual
frame_1.create_text(962, 630, text='Press "Continue" to proceed.', font=("Elephant", 8))
b_2 = ttk.Button(win, text="User Manual", style="C.TButton", command=openManual)
b_2.place(x=875, y=655)

cap = cv2.VideoCapture(0)

w = 600
h = 430
label1 = tk.Label(frame_1, width=w, height=h)
label1.place(x=250, y=150)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

""" Main function for displaying the video (from OpenCV) on the screen """
def select_img():
    _, img = cap.read()
    img = cv2.resize(img, (w, h))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    finalImage = ImageTk.PhotoImage(image)
    label1.configure(image=finalImage)
    label1.image = finalImage
    win.after(1, select_img)


select_img()
win.mainloop()
