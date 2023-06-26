import cv2
import numpy as np
import yaml
from tqdm import tqdm
import os
import sys
import pickle
import logging
from constants import *
from functions import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from confidential_data_manager import load_video_data, initialize_data
        

logging.basicConfig(level=logging.DEBUG)  # Log all messages, DEBUG < INFO < WARNING < ERROR < CRITICAL

# Load the YAML file
with open(CONFIG_YAML, 'r') as file:
    config_data = yaml.safe_load(file)

# Access the relevant information
branding = config_data['production']['branding']
names = config_data['production']['names']
weight_class = config_data['production']['weight-class']
round_len = config_data['production']['rnd-length-s']
frame_rate = config_data['production']['frame-rate']
rounds_scheduled = config_data['production']['rounds-scheduled']

# Print sample name
print('Names:', names)
print('Weight class:', weight_class)

video_data = load_video_data()
rounds, frame_offset, video_dataframes_roi, view_angles, red_values, blue_values = initialize_data()
cameras = load_videos(CAPTURE_FILES)


for cam_key in cameras:
    cameras[cam_key].set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)


if "cam_00" in cameras:
    cameras["cam_00"].set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
if "cam_01" in cameras and "cam_01" in frame_offset:
    cameras["cam_01"].set(cv2.CAP_PROP_POS_FRAMES, max(0, FRAME_INDEX - frame_offset["cam_01"]))
if "cam_02" in cameras and "cam_02" in frame_offset:
    cameras["cam_02"].set(cv2.CAP_PROP_POS_FRAMES, max(0, FRAME_INDEX - frame_offset["cam_02"]))


# Initialize other variables
frame_counts = []
if "cam_00" in cameras:
    frame_count_00 = int(cameras["cam_00"].get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counts.append(frame_count_00)
if "cam_01" in cameras and "cam_01" in frame_offset:
    frame_count_01 = int(cameras["cam_01"].get(cv2.CAP_PROP_FRAME_COUNT)) + frame_offset["cam_01"]
    frame_counts.append(frame_count_01)
if "cam_02" in cameras and "cam_02" in frame_offset:
    frame_count_02 = int(cameras["cam_02"].get(cv2.CAP_PROP_FRAME_COUNT)) + frame_offset["cam_02"]
    frame_counts.append(frame_count_02)

frame_count = min(frame_counts) if frame_counts else 0


frame_rate = int(cameras["cam_00"].get(cv2.CAP_PROP_FPS))
delay = int(1000 / frame_rate)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, frame_rate, (DESIRED_WIDTH, DESIRED_HEIGHT))

best_view_angles = get_best_view_angles(view_angles, WINDOW_SIZE, WEIGHT_FACTOR)

current_frame_index = FRAME_INDEX
prev_ema_roi = None


highlight_segments = get_highlight_sequences(red_values, blue_values)
highlight_index = 0
highlight_displayed = False
end_frame_first_round = rounds[next(iter(rounds))][1]
ret = True  # Initialize ret to True
delay_counter = 0  # Counter for frames since start of break
highlight_cap = cv2.VideoCapture('highlight_frames.mp4')
highlight_dir = 'highlights'
highlight_files = sorted(os.listdir(highlight_dir))

# Reset video streams to the saved position
for camera in cameras.values():
    camera.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)

if highlight_segments:
    current_highlight_frame = highlight_segments[0][0]  # Initialize to the start frame of the first highlight
else:
    current_highlight_frame = None

highlight_frame_counter = 0
highlight_index = 0
highlight_cap = None

if LOAD_HIGHLIGHTS:
    highlight_roi = load_highlight_frames(cameras, highlight_segments, best_view_angles, video_dataframes_roi, ALPHA, frame_rate)
    with open('./viz/highlight_roi.pkl', 'wb') as f:
        pickle.dump(highlight_roi, f)
else:
    with open('./viz/highlight_roi.pkl', 'rb') as f:
        highlight_roi = pickle.load(f)


for FRAME_INDEX in tqdm(range(FRAME_INDEX, frame_count), initial=FRAME_INDEX, total=frame_count):

    if FRAME_INDEX >= frame_count:
        break

    # Advance the main video stream
    if FRAME_INDEX < len(best_view_angles):
        best_camera = f"cam_0{best_view_angles[FRAME_INDEX]}"
    else:
        print(f"Warning: FRAME_INDEX {FRAME_INDEX} is out of range for the list best_view_angles. Length of best_view_angles: {len(best_view_angles)}")
        # May need sensible default or error action here, if FRAME_INDEX out of range
        best_camera = None  # or any default value that makes sense in your context

    ret, frame = advance_frame_with_best_view(FRAME_INDEX, cameras, frame_offset, best_camera)

    # Determine what to display
    if is_outside_rounds(FRAME_INDEX, rounds) and FRAME_INDEX > end_frame_first_round:
        # Increment delay counter
        delay_counter += 1
        # If we've passed the delay period and there are still highlights left
        if delay_counter > DELAY_FRAMES and highlight_index < len(highlight_segments):
            highlight_displayed = True
            # Determine the current frame of the highlight
            frame_to_process = highlight_segments[highlight_index][0] + highlight_frame_counter // SLOW_MOTION_FACTOR  # Added slow-motion factor here
            if highlight_frame_counter < (highlight_segments[highlight_index][1] - highlight_segments[highlight_index][0]) * SLOW_MOTION_FACTOR:  # Adjusted for slow-motion here
                highlight_frame_counter += 1
            else:
                highlight_frame_counter = 0
                highlight_index += 1
        else:
            highlight_displayed = False
            frame_to_process = FRAME_INDEX
    else:
        highlight_displayed = False
        frame_to_process = FRAME_INDEX
        # Reset delay counter when we re-enter a round
        delay_counter = 0


    # If all highlights shown, return to the main video
    if highlight_index >= len(highlight_segments):
        highlight_displayed = False

    if not ret:
        break

    # Get the ROI data for best camera and crop frame only when highlights are not displayed
    if not highlight_displayed:
        frame_roi_data = video_dataframes_roi[best_camera][frame_to_process].get('roi', [])
        for roi in frame_roi_data:
            x1, y1, x2, y2, _, _ = roi
            prev_ema_roi = apply_ema(prev_ema_roi, [x1, y1, x2, y2], ALPHA)

        # Check if the current frame should show ROI-view or full overview
        if is_roi_view(frame_to_process, rounds) and prev_ema_roi is not None:
            cropped_frame = crop_frame_around_roi(frame, prev_ema_roi, DESIRED_WIDTH, DESIRED_HEIGHT)
        else:
            cropped_frame = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT), interpolation=cv2.INTER_LINEAR)
    else:
        # If we're starting a new highlight or if highlight_cap is None, open new VideoCapture
        if highlight_displayed:
            # If we're starting a new highlight or if highlight_cap is None, open new VideoCapture
            if highlight_frame_counter == 0 or highlight_cap is None:
                if highlight_cap is not None:
                    highlight_cap.release()
                highlight_path = os.path.join(highlight_dir, f'highlight_{highlight_index}.mp4')
                highlight_cap = cv2.VideoCapture(highlight_path)
            
            if highlight_cap is not None:
                num_frames = int(highlight_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
                frame_position = min(highlight_frame_counter // SLOW_MOTION_FACTOR, num_frames - 1)  # Ensure frame position does not exceed total frames
                highlight_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                _, highlight_frame = highlight_cap.read()

            # Check if current frame should show ROI-view or the full overview
            if prev_ema_roi is not None:
                cropped_frame = highlight_frame  # No cropping needed here, it was already done when creating the highlight
            else:
                cropped_frame = cv2.resize(highlight_frame, (DESIRED_WIDTH, DESIRED_HEIGHT), interpolation=cv2.INTER_LINEAR)




    # Add overlay only during rounds
    if not highlight_displayed:
        # Overlay
        transition_frames = 60  # Adjust this value to control the speed of the animation
        cropped_frame = create_overlay(cropped_frame, frame_to_process, rounds, FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, branding, names, weight_class, transition_frames, round_len, frame_rate, rounds_scheduled)

    # Display the frame count
    cv2.putText(cropped_frame, f"Frame: {FRAME_INDEX}", TEXT_POSITION, FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

    # Display the resized frame (toggle)
    cv2.imshow('frame', cropped_frame)

    out.write(cropped_frame)

    # Adjust playback speed (toggle)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break



# Clean up
for camera in ["cam_00", "cam_01", "cam_02"]:
    if camera in cameras:
        cameras[camera].release()

cv2.destroyAllWindows()

out.release()
if highlight_cap is not None:
    highlight_cap.release()

