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

def handle_frame(image):
    global frame_idx
    frame_idx += 1
    batch_idx = frame_idx % 100
    track_pixel = np.mean(image[50:100, 50:100])
    batch.append((batch_idx, track_pixel))
    return frame_idx

def process_frame(origin):
    idx_list, pixels = zip(*batch)
    last_idx = idx_list[-1]
    print(last_idx)
    if last_idx > 0 and last_idx % 2 == 0:
        pixels = np.array(pixels[1:])
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