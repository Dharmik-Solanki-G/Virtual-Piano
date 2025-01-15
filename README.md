# Virtual Piano using Hand Gestures

This project uses computer vision and hand gesture recognition to create a virtual piano that can be played by detecting raised fingers. It utilizes OpenCV for image processing, Mediapipe for hand tracking, and Pygame for sound playback.

## Prerequisites

- Python 3.8.10 (or later)
- Install the required libraries using the following command:

    ```bash
    pip install -r requirements.txt
    ```

## Project Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/Dharmik-Solanki-G/Virtual-Piano.git
    cd Virtual-Piano
    ```

2. Ensure you have a webcam or use a video file for input. You can replace the `assets\video.mp4` with your webcam or another video source.

3. Place your sound files in the `assets/sounds/` folder. Each sound file should correspond to a note (e.g., `a1.wav`, `d1.wav`, etc.).

## Running the Project

To run the virtual piano:

```bash
python virtual_piano.py
