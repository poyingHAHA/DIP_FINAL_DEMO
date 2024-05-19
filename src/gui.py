#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:01:22 2024

@author: lanweiren
"""

import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# GUI介面設定
class BoundingBoxApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Bounding Box Drawer")
        # 按下add按鈕後，crop出來的圖片會存到output這個list裡面
        self.output = []
        self.temp_output = None
        self.fixed_window_width = 800
        self.fixed_window_height = 600

        self.canvas = tk.Canvas(self.master, width=self.fixed_window_width, height=self.fixed_window_height)
        self.canvas.grid(row=0, column=0, columnspan=3)  # 改變了 grid() 的參數
        # self.canvas.pack()

        self.image = None
        self.photo = None
        self.zoom_level = 1.0
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.rect = None

        self.upload_button = tk.Button(self.master, text="Upload Photo", command=self.load_image)
        self.upload_button.grid(row=1, column=0, padx=5, pady=5)  # 改變了 grid() 的參數

        # self.crop_button = tk.Button(self.master, text="Crop Image", command=lambda: self.crop_image(self.start_x, self.start_y, self.end_x, self.end_y))
        # self.crop_button.grid(row=1, column=1, padx=10, pady=10)  # 改變了 grid() 的參數
        
        self.add_button = tk.Button(self.master, text="Add", command=lambda: self.output.append(self.temp_output))
        self.add_button.grid(row=1, column=1, padx=5, pady=5)
        
        self.exit_button = tk.Button(self.master, text="Exit", command=self.exit_app)
        self.exit_button.grid(row=1, column=2, padx=5, pady=5)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.drawing)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-3>", self.on_mouse_wheel_press)
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_wheel_release)
        self.canvas.bind("<B3-Motion>", self.on_mouse_wheel_motion)

        # Variables to track dragging with mouse wheel
        self.prev_x = None
        self.prev_y = None
        self.mouse_wheel_pressed = False
        self.canvas_move_x = 0
        self.canvas_move_y = 0

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(self.image)
            self.fit_image_to_window()
            self.canvas_move_x = 0
            self.canvas_move_y = 0

    def fit_image_to_window(self):
        max_width = self.fixed_window_width
        max_height = self.fixed_window_height
        image_width, image_height = self.image.size
        width_ratio = max_width / image_width
        height_ratio = max_height / image_height
        self.zoom_level = min(width_ratio, height_ratio)
        self.show_image()

    def show_image(self):
        self.zoomed_image = self.image.resize((int(self.image.width * self.zoom_level), int(self.image.height * self.zoom_level)))
        self.photo = ImageTk.PhotoImage(self.zoomed_image)
        self.canvas.delete("all")  # Clear canvas before drawing new image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def start_draw(self, event):
        self.start_x = event.x
        self.start_y = event.y
        # self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='red')

    def drawing(self, event):
        if self.rect:
            self.canvas.delete(self.rect)
        x, y = self.start_x, self.start_y
        x1, y1 = event.x, event.y
        self.rect = self.canvas.create_rectangle(x, y, x1, y1, outline="red")

    def end_draw(self, event):
        x0, y0 = self.start_x, self.start_y
        self.end_x, self.end_y = event.x, event.y
        # print("Start Coordinates: x={}, y={}".format(x0, y0))
        # print("End Coordinates: x={}, y={}".format(self.end_x, self.end_y))
        # Adjust coordinates based on zoom level
        if x0 > self.end_x:
            x0, self.end_x = self.end_x, x0
        if y0 > self.end_y:
            y0, self.end_y = self.end_y, y0
        x = int(x0 / self.zoom_level)
        y = int(y0 / self.zoom_level)
        end_x = int(self.end_x / self.zoom_level)
        end_y = int(self.end_y / self.zoom_level)
        
        # print("Canvas Move Coordinates: x={}, y={}".format(self.canvas_move_x, self.canvas_move_y))
        
        # Adjust coordinates based on canvas movement
        x -= self.canvas_move_x / self.zoom_level
        y -= self.canvas_move_y / self.zoom_level
        end_x -= self.canvas_move_x / self.zoom_level
        end_y -= self.canvas_move_y / self.zoom_level
        
        width = end_x - x
        height = end_y - y
        # print("Bounding Box Coordinates: x={}, y={}, width={}, height={}".format(x, y, width, height))
        # Perform cropping based on adjusted coordinates
        self.crop_image(int(x), int(y), int(end_x), int(end_y))

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_level *= 1.1  # Zoom in
        else:
            self.zoom_level /= 1.1  # Zoom out
        self.show_image()

    def on_mouse_wheel_press(self, event):
        self.mouse_wheel_pressed = True
        self.prev_x = event.x
        self.prev_y = event.y

    def on_mouse_wheel_release(self, event):
        self.mouse_wheel_pressed = False
        self.prev_x = None
        self.prev_y = None

    def on_mouse_wheel_motion(self, event):
        if self.mouse_wheel_pressed:
            delta_x = event.x - self.prev_x
            delta_y = event.y - self.prev_y
            self.canvas_move_x += delta_x
            self.canvas_move_y += delta_y
            self.canvas.move(tk.ALL, delta_x, delta_y)
            self.prev_x = event.x
            self.prev_y = event.y

    def crop_image(self, x, y, x1, y1):
        if self.image:
            cropped_image = self.image.crop((x, y, x1, y1))
            self.temp_output = cropped_image
            cropped_image.show()  # You can display or save the cropped image as required
    
    def exit_app(self):
        self.master.destroy()

def get_matted_watermark():
    root = tk.Tk()
    app = BoundingBoxApp(root)
    root.mainloop()
    return app.output