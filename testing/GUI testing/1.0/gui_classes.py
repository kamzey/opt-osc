import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from processing_functions import *
from camera_functions import *
from plotting_functions import *

class Trace(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.columnconfigure(0, weight=3)
        self.__create_widgets()
    
    def __create_widgets(self):
        # Running trace
        canvas = FigureCanvasTkAgg(fig_trace, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
class Image(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.columnconfigure(0, weight=1)
        self.__create_widgets()

    def __create_widgets(self):
        canvas = FigureCanvasTkAgg(fig_img, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class VisualisationWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Visualisation Window')
        self.columnconfigure(1, weight=4)
        self.columnconfigure(0, weight=1)
        self.__create_widgets()

    def __create_widgets(self):
        image_frame = Image(self)
        image_frame.grid(column=0, row=0)
        trace_frame = Trace(self)
        trace_frame.grid(column=1, row=0)
        