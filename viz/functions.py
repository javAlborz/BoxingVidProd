import cv2
import pandas as pd
import pickle
import json
import numpy as np
import os
from pprint import pprint
from tqdm import tqdm

from constants import *
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from confidential_data_manager import load_video_data, initialize_data

# Load data

def load_videos(capture_files):
    cameras = {}
    for i, file_path in enumerate(capture_files):
        cam_key = f"cam_0{i}"
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            cameras[cam_key] = cap
        else:
            print(f"Failed to open video file: {file_path}")
    return cameras

    
# Process video

def apply_ema(prev_ema, current_coords, alpha):
    if prev_ema is None:
        return current_coords
    else:
        ema_coords = [0, 0, 0, 0]
        for i in range(4):
            ema_coords[i] = (alpha * current_coords[i]) + ((1 - alpha) * prev_ema[i])
        return ema_coords

def crop_frame_around_roi(frame, roi, output_width, output_height, padding=100, y_shift=200):
    x1, y1, x2, y2 = roi
    roi_center_x = int((x1 + x2) / 2)
    roi_center_y = int((y1 + y2) / 2)

    half_width = (output_width + padding * 2) // 2
    half_height = (output_height + padding * 2) // 2

    crop_x1 = max(0, roi_center_x - half_width)
    crop_y1 = max(0, roi_center_y - half_height - y_shift)
    crop_x2 = crop_x1 + output_width + padding * 2
    crop_y2 = crop_y1 + output_height + padding * 2

    if crop_x2 > frame.shape[1]:
        diff_x = crop_x2 - frame.shape[1]
        crop_x1 -= diff_x
        crop_x2 -= diff_x

    if crop_y2 > frame.shape[0]:
        diff_y = crop_y2 - frame.shape[0]
        crop_y1 -= diff_y
        crop_y2 -= diff_y

    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    resized_frame = cv2.resize(cropped_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    return resized_frame

def is_roi_view(frame_index, rounds):
    for round_key in rounds:
        start, end, breaks = rounds[round_key]
        if start <= frame_index <= end:
            # Check if the frame is within an unexpected break
            for break_start, break_end in breaks:
                if break_start <= frame_index <= break_end:
                    return False
            return True
    return False

def read_frame_with_best_view(frame_index, caps, frame_offset, best_view):
    ret, frame = False, None
    if best_view == 0 and frame_index >= max(frame_offset["cam_01"], frame_offset["cam_02"]):
        ret, frame = caps["cam_00"].read()
    elif best_view == 1:
        ret, frame = caps["cam_01"].read()
    elif best_view == 2:
        ret, frame = caps["cam_02"].read()

    return {f"cam_0{best_view}": (ret, frame)}

def get_best_view_angles(view_angles, window_size, weight_factor):
    camera_names = list(view_angles.keys())
    min_length = min(len(view_angles[camera]) for camera in camera_names)

    if len(camera_names) == 1:  # only one camera is available
        return [0] * min_length

    else:  # more than one camera is available
        best_view_angles = []
        prev_best_camera = None

        for i in range(min_length - window_size + 1):
            window_view_angles = [np.sum(view_angles[camera][i:i + window_size]) for camera in camera_names]

            best_camera = np.argmax(window_view_angles)

            if prev_best_camera is not None:
                weighted_view_angles = [0]*len(camera_names)
                weighted_view_angles[prev_best_camera] = weight_factor * window_size
                weighted_view_angles = np.add(weighted_view_angles, window_view_angles)
                best_camera = np.argmax(weighted_view_angles)

            best_view_angles.extend([best_camera] * window_size)
            prev_best_camera = best_camera

        return best_view_angles[:min_length]

def advance_frame_with_best_view(frame_index, caps, frame_offset, best_view):
    ret, frame = {}, {}
    for camera in ["cam_00", "cam_01", "cam_02"]:
        if camera in caps:  # Added this line to check if the camera exists
            if camera == "cam_00" and frame_index < max(frame_offset.get("cam_01", 0), frame_offset.get("cam_02", 0)):
                ret[camera], frame[camera] = False, None
            else:
                ret[camera], frame[camera] = caps[camera].read()

    return ret.get(best_view, False), frame.get(best_view, None)





# Overlay
def create_overlay(frame, frame_index, rounds, font, font_scale, font_color, font_thickness, branding, names, weight_class, transition_frames, round_len, frame_rate, rounds_scheduled):
    in_round = False
    start, end = 0, 0
    previous_end = 0
    next_start = float('inf')
    sorted_rounds_keys = sorted(rounds.keys())
    current_round = 0

    for round_key in sorted_rounds_keys:
        start, end, _ = rounds[round_key]
        current_round += 1
        if start <= frame_index <= end:
            in_round = True
            break
        elif start > frame_index:
            next_start = start
            break
        else:
            previous_end = end

    overlay_height = int(frame.shape[0] * 0.17)
    overlay_width = int(frame.shape[1] * 0.33)
    horizontal_offset = int(frame.shape[1] * 0.05)
    transition_step = overlay_height / transition_frames

    # Create the overlay
    overlay = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)
    overlay_color = (50, 50, 50)
    overlay[:] = overlay_color

    text_x = 100  # moves the text to the right
    text_y = 30

    # Display the round number in the overlay
    round_text = f"{current_round} OF {rounds_scheduled}"
    cv2.putText(overlay, round_text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    # Display the names in red or blue
    red_name = names['red']
    blue_name = names['blue']
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)

    # Define the outline color and thickness
    outline_color = (255, 255, 255)
    outline_thickness = font_thickness + 2

    # Draw the outline for the red name
    cv2.putText(overlay, red_name, (text_x, text_y + 40), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)

    # Draw the red name
    cv2.putText(overlay, red_name, (text_x, text_y + 40), font, font_scale, red_color, font_thickness, cv2.LINE_AA)

    # Draw the outline for the blue name
    cv2.putText(overlay, blue_name, (text_x, text_y + 80), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)

    # Draw the blue name
    cv2.putText(overlay, blue_name, (text_x, text_y + 80), font, font_scale, blue_color, font_thickness, cv2.LINE_AA)

    # Load the image
    photo = cv2.imread('dtu.png')
    # Resize the photo to fit within the overlay
    photo = cv2.resize(photo, (80, 80))
    photo_top_left = (0, 0)
    overlay[photo_top_left[1]:photo_top_left[1]+photo.shape[0], photo_top_left[0]:photo_top_left[0]+photo.shape[1]] = photo

    # Calculate remaining time in the round (min:sec)
    round_start_time = start / frame_rate
    elapsed_time_in_round = (frame_index / frame_rate) - round_start_time
    remaining_time_in_round = round_len - elapsed_time_in_round
    minutes, seconds = divmod(int(remaining_time_in_round), 60)
    timer_text = f"{minutes}:{seconds:02}"
    # Set timer color
    timer_color = (255, 255, 255)  
    # Set timer position
    timer_x = 5  # position timer to the left of other text
    timer_y = 110  # adjust as needed
    # Add timer text to the overlay
    cv2.putText(overlay, timer_text, (timer_x, timer_y), font, font_scale, timer_color, font_thickness)
    timer_box_color = (100, 100, 100) 
    timer_box_thickness = 2  
    timer_box_top_left = (timer_x - 10, timer_y - 110) 
    timer_box_bottom_right = (timer_x + 75, timer_y + 10)
    cv2.rectangle(overlay, timer_box_top_left, timer_box_bottom_right, timer_box_color, timer_box_thickness)

    # The vertical offset from the bottom of the screen
    vertical_offset = int(frame.shape[0] * 0.1)


    # We are in round, handle transition in at round start
    if in_round and start < frame_index <= start + transition_frames:
        slide_in_position = frame.shape[0] - int((frame_index - start) * transition_step) - vertical_offset
        overlay_slice_height = min(overlay_height, frame.shape[0] - slide_in_position)
        frame[slide_in_position : slide_in_position + overlay_slice_height, horizontal_offset:overlay_width + horizontal_offset] = cv2.addWeighted(frame[slide_in_position : slide_in_position + overlay_slice_height, horizontal_offset:overlay_width + horizontal_offset], 0.5, overlay[:overlay_slice_height, :], 0.5, 0)

    # We are in round but past transition in, show the full overlay
    elif in_round and frame_index > start + transition_frames:
        frame[-overlay_height - vertical_offset: -vertical_offset, horizontal_offset:overlay_width + horizontal_offset] = cv2.addWeighted(frame[-overlay_height - vertical_offset: -vertical_offset, horizontal_offset:overlay_width + horizontal_offset], 0.5, overlay, 0.5, 0)

    # We are at round end, handle transition out
    elif not in_round and previous_end < frame_index <= previous_end + transition_frames:
        slide_out_position = int((frame_index - previous_end) * transition_step)
        frame_position = frame.shape[0] - vertical_offset - overlay_height + slide_out_position

        if frame_position < frame.shape[0]:
            overlay_slice_height = min(overlay_height, frame.shape[0] - frame_position)
            frame_bottom = min(frame.shape[0], frame_position + overlay_height)
            frame[frame_position:frame_bottom, horizontal_offset:overlay_width + horizontal_offset] = cv2.addWeighted(frame[frame_position:frame_bottom, horizontal_offset:overlay_width + horizontal_offset], 0.5, overlay[:overlay_slice_height,:], 0.5, 0)

    return frame



#Highlights
def is_outside_rounds(frame_index, rounds):
    for round_key in rounds:
        start, end, breaks = rounds[round_key]
        if start <= frame_index <= end:
            # Check if the frame is within an unexpected break
            for break_start, break_end in breaks:
                if break_start <= frame_index <= break_end:
                    return False
            return False  # Inside an interval
    return True  # Outside all intervals


def get_highlight_sequences(red_values, blue_values, buffer_frames=50):
    def get_peak_sequence(values):
        peak_frame = np.argmax(values)
        start = max(0, peak_frame - buffer_frames)
        end = min(len(values) - 1, peak_frame + buffer_frames)
        return [start, end]

    red_sequences = []
    for cam in red_values:
        red_sequences.append(get_peak_sequence(red_values[cam]))
    longest_red_sequence = max(red_sequences, key=lambda seq: seq[1] - seq[0])

    blue_sequences = []
    for cam in blue_values:
        blue_sequences.append(get_peak_sequence(blue_values[cam]))
    longest_blue_sequence = max(blue_sequences, key=lambda seq: seq[1] - seq[0])

    return longest_red_sequence, longest_blue_sequence

def load_highlight_frames(cameras, highlight_sequences, best_view_angles, video_dataframes_roi, ALPHA, frame_rate, output_dir='highlights'):
    os.makedirs(output_dir, exist_ok=True)
    highlight_roi = {}
    prev_ema_roi = None  

    total_frames = sum(sequence[1] - sequence[0] + 1 for sequence in highlight_sequences)
    progress_bar = tqdm(total=total_frames, desc="Loading highlight frames")

    for sequence_index, sequence in enumerate(highlight_sequences):
        highlight_writer = cv2.VideoWriter(os.path.join(output_dir, f'highlight_{sequence_index}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (DESIRED_WIDTH, DESIRED_HEIGHT))
        for frame in range(sequence[0], sequence[1] + 1):
            best_camera = f"cam_0{best_view_angles[frame]}"
            cameras[best_camera].set(cv2.CAP_PROP_POS_FRAMES, frame)
            _, loaded_frame = cameras[best_camera].read()

            # Update ROI for highlight frame
            frame_roi_data = video_dataframes_roi[best_camera][frame].get('roi', [])
            for roi in frame_roi_data:
                x1, y1, x2, y2, _, _ = roi
                prev_ema_roi = apply_ema(prev_ema_roi, [x1, y1, x2, y2], ALPHA)
            highlight_roi[frame] = prev_ema_roi

            # Crop to the ROI and resize
            if prev_ema_roi is not None:
                loaded_frame = crop_frame_around_roi(loaded_frame, prev_ema_roi, DESIRED_WIDTH, DESIRED_HEIGHT)
            else:
                loaded_frame = cv2.resize(loaded_frame, (DESIRED_WIDTH, DESIRED_HEIGHT), interpolation=cv2.INTER_LINEAR)

            # Write the frame to the current highlight video
            highlight_writer.write(loaded_frame)

            # Update the progress bar after each frame
            progress_bar.update()

        # Release the writer after finishing each sequence
        highlight_writer.release()

    progress_bar.close()
    return highlight_roi



