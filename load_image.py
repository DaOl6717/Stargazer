import rawpy
import numpy as np

def load_raw(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess()
    return rgb