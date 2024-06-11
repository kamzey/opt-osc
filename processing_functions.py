### Processing functions ###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from collections import deque

batch_size = 100
batch = deque(maxlen=batch_size)
batch.append((0, 0))

origin = (1000 // 2) - 1

frame_idx = 0

def handle_frame(image, selections): # selections a LIST of rois
    global frame_idx
    frame_idx += 1
    batch_idx = frame_idx % 100
    def extract_rois(roi):
        pixels = np.mean(image[roi])
        return pixels
    rois = map(extract_rois, selections) # feed each roi to function
    batch.append((batch_idx, rois)) # rois now a LIST of LISTs
    return frame_idx

def process_frame(origin):
    idx_list, rois = zip(*batch)
    last_idx = idx_list[-1]
    print(last_idx)
    def process(roi):
        if last_idx > 0 and last_idx % 2 == 0:
            pixels = np.array(roi[1:])
            blue = pixels[:last_idx-1:2].astype(float)
            green = pixels[1:last_idx:2].astype(float)
            blue /= ndimage.uniform_filter(blue, size=1000, mode='nearest',
                                            origin=origin)
            green /= ndimage.uniform_filter(green, size=1000, mode='nearest',
                                            origin=origin)
            proc_points = blue/green
            print(proc_points[-1])
            # idx = int(last_idx / 2)
    #        if proc_points.size != 0:
            return proc_points[-1]
    proc_points_all = map(process, rois)
    return proc_points_all

