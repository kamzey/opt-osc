import tkinter as tk
import matplotlib.pyplot as plt
from processing_functions import *

fig_trace, ax_trace = plt.subplots()
fig_img, ax_img = plt.subplots()
# running_trace, = ax_trace.plot([], [], color='blue')
trace_points = deque(maxlen=1000)

def update_trace(frames=None):
    while frames < 1000:
        ax_trace.plot(range(len(trace_points)), trace_points)
        fig_trace.canvas.draw()
        return ax_trace, 

def update_image(image):
    ax_img.imshow(image)
    fig_img.canvas.draw()