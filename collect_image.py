import cv2
import tkinter as tk
from tkinter import filedialog

def collect_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    # Open a file dialog to select an image
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    # Read the selected image
    image = cv2.imread(image_path)

    return image
