https://asl-auto-detection.streamlit.app

Real-time ASL Recognition & Sentence Builder
This is a computer vision application that translates American Sign Language (ASL) into text in real-time, capable of constructing coherent sentences from continuous hand gestures.

Features

Real-Time Detection: Detects and classifies hand gestures via webcam with low latency using YOLOv8 models.

Sentence Builder: Automatically constructs sentences from detected letters using a smart stability algorithm.

Live Feedback: Provides visual bounding boxes and confidence scores directly on the video feed.

Real-time ASL Recognition - Application Flow Documentation

1. Overview The ASL Recognition System is a Streamlit-based application designed to bridge communication gaps for the deaf community. It combines YOLOv8 Object Detection, Image Classification, and State Management Logic to translate visual gestures into structured text output in real-time.

2. High-Level Architecture The application consists of four main pillars:

Frontend: Built with Streamlit and streamlit-webrtc for handling live video streams.

Detection Layer: A generic YOLOv8 model (runs/detect/...) to localize hands in the frame.

Classification Layer: A specialized YOLOv8 classifier (runs/classify/...) to identify specific ASL letters.

Logic Layer: A custom SharedState class to manage sentence construction and filter noise.

3. Step-by-Step Data Flow

Step 1: Video Capture & Pre-processing

Action: User enables the webcam via the web interface.

Module: streamlit_app.py -> webrtc_streamer

Process: Captures video frames in real-time and converts them to a NumPy array for processing.

Step 2: Object Detection (Localization)

Action: The system scans the frame to find hand position.

Module: streamlit_app.py -> VideoProcessor.recv()

Process: Passes the frame to the YOLOv8 Detection Model. It returns bounding box coordinates, cropping the image to focus solely on the hand region.

Step 3: Gesture Classification

Action: The cropped hand image is analyzed to determine the letter.

Module: streamlit_app.py -> classifier.predict()

Process: The cropped image is fed into the YOLOv8 Classification Model, which outputs the predicted letter (Class ID) and a confidence score.

Step 4: Sentence Logic & Stabilization

Action: The system determines if the gesture is intentional or noise.

Module: streamlit_app.py -> SharedState Logic

Process:

Implements a 2-second stability threshold: The same letter must be held continuously for 2 seconds to be registered.

Updates the progress bar on the UI to show "Holding" status.

Appends valid letters to the constructed sentence string.

Step 5: User Interface Update

Action: Displays the result to the user.

Process: Overlays bounding boxes and predicted text on the video stream and updates the "Constructed Sentence" text area dynamically.
