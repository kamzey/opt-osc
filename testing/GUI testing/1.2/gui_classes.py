import tkinter as tk
import tifffile as tf
from tkinter import ttk
from PIL import ImageTk
from PIL import Image as ImagePIL
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pycromanager
import numpy as np
#from processing_functions import *
#from camera_functions import *
#from plotting_functions import *
from cellpose import models, plot, io
from threading import Thread
from pycromanager import Core, Acquisition, multi_d_acquisition_events
from matplotlib.animation import FuncAnimation
from collections import deque
from scipy import ndimage
import datetime
import os
import itertools

class Trace(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.columnconfigure(0, weight=3)
        self.fig_trace, self.ax_trace = plt.subplots(len(container.rois), 1) 
        self.trace_points = itertools.repeat(deque(maxlen=1000), len(container.rois))
        self.__create_widgets()
    
    def update_trace(self, frames=None):
        def individual_traces(ax, trace):
            while frames < 1000:
                ax.plot(range(len(trace)), trace)
                self.fig_trace.canvas.draw()
                return ax, 
        traces = map(individual_traces, self.ax_trace, self.trace_points)

    def __create_widgets(self):
        canvas = FigureCanvasTkAgg(self.fig_trace, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
class Image(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        # self.image = image
        self.columnconfigure(0, weight=1)
        self.fig_img, self.ax_img = plt.subplots()
        self.__create_widgets()

    def update_image(self, image):
        self.ax_img.imshow(image)
        self.fig_img.canvas.draw()

    def __create_widgets(self):
        canvas = FigureCanvasTkAgg(self.fig_img, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class Segment(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.rowconfigure(0, weight=5)
        self.__create_widgets()
        self.image_path = r"data_from_paper.ome.tif"
        self.selections = []
        self.make_segment_image()

    def make_segment_image(self):
        try:
            img = tf.imread(self.image_path)
            self.image = img[0, :, :]
        except:
            print("Image cannot be opened")
        self.image_np = np.array(self.image)
        # Segment the image
        model = models.Cellpose(model_type='cyto')
        self.masks, flows, styles, diams = model.eval(self.image, diameter=30, channels=[0,0])
        # Convert masks to an RGB overlay
        print(type(self.masks))
        print(self.masks)
        print(f"Original image min: {self.image_np.min()}, max: {self.image_np.max()}")

        # Normalize the image data to the range 0-255 while maintaining the original contrast
        image_min = self.image_np.min()
        image_max = self.image_np.max()

        self.image_np_normalized = ((self.image_np - image_min) / (image_max - image_min) * 255).astype(np.uint8)

        # Print min and max values after normalization for debugging
        print(f"Normalized image min: {self.image_np_normalized.min()}, max: {self.image_np_normalized.max()}")

        # Print min and max values after normalization for debugging
        print(f"Normalized image min: {self.image_np_normalized.min()}, max: {self.image_np_normalized.max()}")
        pil_img = ImagePIL.fromarray(self.image_np_normalized)
        overlay = plot.mask_overlay(self.image_np, self.masks)
        overlay_image = ImagePIL.fromarray((overlay * 255).astype(np.uint8))
        self.overlay_image = overlay_image
        self.tk_overlay = ImageTk.PhotoImage(overlay_image)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        # Update canvas with the new overlay image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.create_rois_from_masks()
        self.canvas.bind("<Button-1>", self.click_events)

    def create_rois_from_masks(self):
        unique_masks = np.unique(self.masks)
        clickable_areas = []
        for mask_val in unique_masks:
            if mask_val == 0:
                continue  # Skip the background
            y, x = np.where(self.masks == mask_val)
            if len(x) == 0 or len(y) == 0:
                continue
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            boundaries = (min_x, max_x, min_y, max_y)
            clickable_areas.append(boundaries)
        self.rois = clickable_areas

    def click_events(self, event):
        for roi in self.rois:
            (x1, x2, y1, y2) = roi
            if x1 < event.x < x2 and y1 < event.y < y2:
                self.on_button_click(event, roi)

    def on_button_click(self, event, roi):
        self.selections.append(roi) # for feeding into processing algorithm
        (x1, x2, y1, y2) = roi
        self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)

    def save_overlay(self):
        if self.overlay_image:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Get the current working directory
            current_directory = os.getcwd()
            # Create the full save path
            overlay_path = os.path.join(current_directory, f"overlay_{timestamp}.png")
            self.overlay_image.save(overlay_path)
            print(f"Overlay saved to {overlay_path}")
        else:
            print("Overlay image is not available")

    def __create_widgets(self):
        # Initialize the canvas
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0, sticky='nsew')

class Next(ttk.Frame):
    def __init__(self, master, rois):
        super().__init__(master)
        self.master = master
        self.rois = rois
        self.__create_widgets()
    
    def new_window(self):
        self.NewWindow = VisualisationWindow(self.rois)
        self.master.destroy()
    
    def __create_widgets(self):
        button = tk.Button(self, text="Next",
                           command=self.new_window)
        button.pack()

class SegmentationWindow(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title("Segmentation Window")
        self.rowconfigure(0, weight=5)
        self.rowconfigure(1, weight=1)
        self.__create_widgets()

    def __create_widgets(self):
        segment_frame = Segment(self)
        segment_frame.grid(column=0, row=0)
        loading_frame = Next(self, segment_frame.rois)
        loading_frame.grid(column=0, row=1)

class VisualisationWindow(tk.Toplevel):
    def __init__(self, selections):
        super().__init__()
        self.title('Visualisation Window')
        self.columnconfigure(1, weight=4)
        self.columnconfigure(0, weight=1)
        self.batch = deque(maxlen=100)
        self.batch.append((0, 0))
        self.origin = (1000 // 2) - 1
        self.rois = selections
        self.frame_idx = 0
        self.__create_widgets()
        self.acquisition()
    
    def acquisition(self):
        with Acquisition(directory=r'C:\Users\aysez\Documents\optical oscilloscope\debug', name="demo_acq", image_process_fn=self.image_process_fn) as acq:
        # Generate acquisition events
            events = multi_d_acquisition_events(
                num_time_points=500,
                time_interval_s = 1
            )
            # Start image acquisition
            acq.acquire(events)
            ani = FuncAnimation(self.trace_frame.fig_trace, self.trace_frame.update_trace, interval=500, blit=True)

    def image_process_fn(self, image, metadata):
        self.handle_frame(image)
        self.image_frame.update_image(image)
        try:
            datapoints = self.process_frame()
            if datapoints is not None:
                x = list(map(self.update_data, self.trace_frame.trace_points, datapoints))
        finally:
            return image, metadata    
    
    def update_data(self, trace, data):
        trace.append(data)

    def handle_frame(self, image): # selections a LIST of rois
        self.frame_idx += 1
        batch_idx = self.frame_idx % 100
        def extract_rois(roi):
            pixels = np.mean(image[roi])
            return pixels
        rois = map(extract_rois, self.rois) # feed each roi to function
        self.batch.append((batch_idx, rois)) # rois now a LIST of LISTs
        return self.frame_idx

    def process_frame(self):
        idx_list, rois = zip(*self.batch)
        last_idx = idx_list[-1]
        print(last_idx)
        def process(roi):
            if last_idx > 0 and last_idx % 2 == 0:
                pixels = np.array(roi[1:])
                blue = pixels[:last_idx-1:2].astype(float)
                green = pixels[1:last_idx:2].astype(float)
                blue /= ndimage.uniform_filter(blue, size=1000, mode='nearest',
                                                origin=self.origin)
                green /= ndimage.uniform_filter(green, size=1000, mode='nearest',
                                                origin=self.origin)
                proc_points = blue/green
                print(proc_points[-1])
                # idx = int(last_idx / 2)
        #        if proc_points.size != 0:
                return proc_points[-1]
        proc_points_all = map(process, rois)
        return proc_points_all


    def __create_widgets(self):
        self.image_frame = Image(self)
        self.image_frame.grid(column=0, row=0)
        self.trace_frame = Trace(self)
        self.trace_frame.grid(column=1, row=0)
        
        
app = SegmentationWindow()
app.mainloop()