### Camera Connection Functions ###

import pycromanager
from processing_functions import *
from plotting_functions import *


def establish_connection():
    return pycromanager.bridge.CMMCore() # GLOBAL variable

def initialise_camera(core):
    device_name = "Device Name"
    core.initializeDevice(device_name)

def pre_segmentation_frame(core):
    core.snapImage()
    return core.getImage()

def image_process_fn(image, metadata):
    handle_frame(image)
    # add image overlay
    update_image(image)
    try:
        datapoint = process_frame(origin)
        if datapoint is not None:
            trace_points.append(datapoint)
    finally:
        return image, metadata
