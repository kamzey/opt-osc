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
from cellpose import models, plot
from threading import Thread
from pycromanager import Acquisition, multi_d_acquisition_events
from matplotlib.animation import FuncAnimation
from collections import deque

class Trace(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.columnconfigure(0, weight=3)
        self.fig_trace, self.ax_trace = plt.subplots() # add length
        self.trace_points = deque(maxlen=1000)
        self.__create_widgets()
    
    def update_trace(self, frames=None):
        while frames < 1000:
            self.ax_trace.plot(range(len(self.trace_points)), self.trace_points)
            self.fig_trace.canvas.draw()
            return self.ax_trace, 

    def __create_widgets(self):
        canvas = FigureCanvasTkAgg(self.fig_trace, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
class Image(ttk.Frame):
    def __init__(self, container, image):
        super().__init__(container)
        self.image = image
        self.columnconfigure(0, weight=1)
        self.fig_img, self.ax_img = plt.subplots()
        self.__create_widgets()

    def update_image(self):
        self.ax_img.imshow(self.image)
        self.fig_img.canvas.draw()

    def __create_widgets(self):
        canvas = FigureCanvasTkAgg(fig_img, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class Segment(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.rowconfigure(0, weight=5)
        self.__create_widgets()
        self.image_path = r"C:\Users\aysez\Downloads\data_from_paper.ome.tif"
        self.rois = []
        self.segment()

    def segment(self):
        try:
            self.image = tf.imread(self.image_path)
        except:
            print("Image cannot be opened")
        self.image_np = np.array(self.image)
        print("Image min value:", self.image_np.min())
        print("Image max value:", self.image_np.max())
        # Segment the image
        model = models.Cellpose(gpu=True, model_type='cyto')
        self.masks, flows, styles, diams = model.eval(self.image, diameter=30, channels=[0,0])
        # Convert masks to an RGB overlay
        overlay = plot.mask_overlay(self.image_np, self.masks)
        print("Overlay min value:", overlay.min())
        print("Overlay max value:", overlay.max())
        overlay_image = ImagePIL.fromarray((overlay * 255).astype(np.uint8))
        print(overlay_image.shape)
        self.tk_overlay = ImageTk.PhotoImage(overlay_image)
        # Update canvas with the new overlay image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_overlay)
        self.create_buttons_from_masks()
    
    def create_buttons_from_masks(self):
        unique_masks = np.unique(self.masks)
        for mask_val in unique_masks:
            if mask_val == 0:
                continue  # Skip the background
            y, x = np.where(self.masks == mask_val)
            if len(x) == 0 or len(y) == 0:
                continue
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            center_x, center_y = int(np.mean(x)), int(np.mean(y))
            boundaries = (min_x, max_x, min_y, max_y)
            button = tk.Button(self, text=f"{mask_val}", 
                               command=lambda b=boundaries: self.on_button_click(b))
            self.canvas.create_window(center_x, center_y, window=button)
    
    def on_button_click(self, boundaries):
        self.rois.append(boundaries) # for feeding into processing algorithm

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
        self.__create_widgets()
        self.acquisition()
    
    def acquisition(self):
        with Acquisition(directory=r'C:\Users\aysez\Documents\optical oscilloscope\debug', name="zelux_acq", image_process_fn=self.image_process_fn) as acq:
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
            datapoint = self.process_frame(self.origin)
            if datapoint is not None:
                self.trace_frame.trace_points.append(datapoint)
        finally:
            return image, metadata    
    
    def handle_frame(self, image): # selections a LIST of rois
        global frame_idx
        frame_idx += 1
        batch_idx = frame_idx % 100
        def extract_rois(roi):
            pixels = np.mean(image[roi])
            return pixels
        rois = map(extract_rois, self.rois) # feed each roi to function
        self.batch.append((batch_idx, rois)) # rois now a LIST of LISTs
        return frame_idx

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