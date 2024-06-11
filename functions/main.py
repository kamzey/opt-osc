import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import pycromanager
from pycromanager import Acquisition, multi_d_acquisition_events
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from processing_functions import *
from camera_functions import *
from plotting_functions import *
from gui_classes import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from collections import deque

save_dir = r'C:\Users\aysez\OneDrive\Documents\cam'

with Acquisition(directory=save_dir, name="zelux_acq", image_process_fn=image_process_fn) as acq:
    # Generate acquisition events
    events = multi_d_acquisition_events(
        num_time_points=500,
        time_interval_s = 1
    )
    # Start image acquisition
    acq.acquire(events)
    app = VisualisationWindow()
    ani = FuncAnimation(fig_trace, update_trace, frames=1000, interval=500, blit=True)
    app.mainloop()

