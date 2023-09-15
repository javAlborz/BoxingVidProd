# AutoVidProd

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Video Demo](#video-demo)
- [Contributing](#contributing)
- [License](#license)

## Introduction
AutoVidProd is a software suite designed to streamline the production process of videos by automating tasks such as camera angle selection, highlights extraction, and overlay application. Utilizing the output of the Jabbr.ai DeepStrike model, you can achieve studio-quality production without manual intervention.

## Features
1. **Camera Angle Selection**: Automatically chooses the best camera angle based on the region of interest (ROI) for each frame.
2. **Highlights Extraction**: Identifies and extracts highlights from the video footage for concise and engaging summaries.
3. **Region of Interest (ROI) Tracking**: Smooth tracking of ROIs.
4. **Overlay Application**: Dynamically adds overlays during specified segments of the video.
5. **Customizable Configuration**: Adjust settings and parameters to tailor the production process to specific needs.
6. **Multi-Camera Support**: Seamlessly switch between multiple camera feeds to get the best view.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/javAlborz/BoxingVidProd

2. Navigate to the project directory:
    cd BoxingVidProd

3. Install the required Python packages:
    pip install -r requirements.txt

## Usage
1. Set your desired video path in the `constants.py` file.

Before running AutoVidProd, ensure you have the necessary configuration files set up:

`time_stamps.json`

This file provides crucial timestamp information for your video:

- **Rounds**: Specifies the start and end frames for each round, along with unexpected break intervals within the round.
- **Sync**: Contains frame offsets essential for synchronizing multiple camera views.
- **Win-Declaration**: Denotes the frame where the winner is declared.

To generate the rounds and sync parameters, refer to:
- [MultiVidSynch](https://github.com/javAlborz/MultiVidSynch) - For synchronizing multi-camera views.
- [RoundBreakDetector](https://github.com/javAlborz/RoundBreakDetector) - For deriving round timestamps.


`config.yaml`

This YAML file lets you tailor the video production parameters to your preferences


2. Run the main script:    
    python viz/main.py --frame_index [your-desired-frame-index]


## Video Demo

Showcasing a side-by-side comparison between the raw camera streams and the enhanced output from AutoVidProd.

| Original Camera Streams | AutoVidProd Output |
|:-----------------------:|:------------------:|
| [![Original Camera Streams](http://img.youtube.com/vi/XWQKIbC_E2Q/0.jpg)](https://www.youtube.com/watch?v=XWQKIbC_E2Q "Original Camera Streams") | [![AutoVidProd Output](http://img.youtube.com/vi/knAk3Bfg11Y/0.jpg)](http://www.youtube.com/watch?v=knAk3Bfg11Y "AutoVidProd Output") |
| [![Original Camera Streams](http://img.youtube.com/vi/oZl95mU2DrE/0.jpg)](https://www.youtube.com/watch?v=oZl95mU2DrE "Original Camera Streams") | [![AutoVidProd Output](http://img.youtube.com/vi/lD1YHWC7lzo/0.jpg)](http://www.youtube.com/watch?v=lD1YHWC7lzo "AutoVidProd Output") |




## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
