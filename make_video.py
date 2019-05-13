import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from skimage import io

import matplotlib.animation as ani
from IPython.display import HTML
import matplotlib

source_dir = Path('./data/1/test/test_img')
target_dir = Path('./results/updated/test_latest/refined_images')

source_img_paths = sorted(source_dir.iterdir())
# target_synth_paths = sorted(target_dir.glob('*synthesized*'))
target_synth_paths = sorted(target_dir.glob('*.png'))

# cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1024,512))
idx = 0
for target, source in zip(target_synth_paths, source_img_paths):
    target_img = cv2.imread(str(target))
    source_img = cv2.imread(str(source))
    if idx < 130:
    	target_img = np.zeros(target_img.shape).astype(np.uint8)
    if idx > 700:
    	break

    frame = np.concatenate((source_img, target_img),axis = 1)
    # print(frame.shape)
    out.write(frame)
    idx += 1

# Release everything if job is finished
out.release()