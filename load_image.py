import rawpy
import numpy as np

def load_raw(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1)
        )
    return rgb