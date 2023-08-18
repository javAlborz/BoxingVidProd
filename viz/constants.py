import cv2
import pandas as pd
import pickle
import json
import numpy as np
import os
from pprint import pprint
import argparse


# Input arguments
parser = argparse.ArgumentParser(description='Integers.')
parser.add_argument('--frame_index', type=int, default=0, help='an integer for the frame index')
args = parser.parse_args()

# Config
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
WINDOW_SIZE = 10
TEXT_POSITION = (10, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2

FRAME_INDEX = args.frame_index #6900, 8550 

# ROI, CAM selection and highlights
ALPHA = 0.03  # roi
WEIGHT_FACTOR = 0.1  # view
WINDOW_SIZE= 12

SLOW_MOTION_FACTOR = 2  # Display each frame this many times
DELAY_FRAMES = 200  # Number of frames to delay before starting highlights
LOAD_HIGHLIGHTS = False 

# File paths
BASE_DIR = "./sample_alborz_3/00085_liu_xudong_vs_nick_konstantin_mtch_1"
TIMESTAMPS_JSON = os.path.join(BASE_DIR, "ground_truth.json")
CONFIG_YAML = os.path.join(BASE_DIR, "config.yaml")

CAPTURE_FILES = [
    os.path.join(BASE_DIR, "original_cam_00.mp4"),
    os.path.join(BASE_DIR, "original_cam_01.mp4"),
    os.path.join(BASE_DIR, "original_cam_02.mp4")
]
OUTPUT_FILE = "./output.mp4"


