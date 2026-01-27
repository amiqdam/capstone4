import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import threading
from collections import deque
from streamlit_autorefresh import st_autorefresh

# Page config
st.set_page_config(
    page_title="ASL Detector",
    page_icon="🤟",
    layout="wide"
)

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_PATH = os.path.join(BASE_DIR, "runs/detect/model_2/weights/best.pt")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "runs/classify/cls_model_3/weights/best.pt")

# Load models (cached)
@st.cache_resource
def load_models():
    try:
        detector = YOLO(DETECTOR_PATH)
        classifier = YOLO(CLASSIFIER_PATH)
        return detector, classifier
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

detector, classifier = load_models()

# Shared state for cross-thread communication
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detected_letter = None
        self.detected_confidence = 0.0
        self.sentence = ""
        self.hold_progress = 0.0
        self.last_appended_letter = None
        self.hold_time = 2.0
        self.sentence_mode = False
    
    def update_detection(self, letter, confidence, hold_progress=0.0):
        with self.lock:
            self.detected_letter = letter
            self.detected_confidence = confidence
            self.hold_progress = hold_progress
    
    def append_to_sentence(self, letter):
        with self.lock:
            self.sentence += letter
            self.last_appended_letter = letter
    
    def get_sentence(self):
        with self.lock:
            return self.sentence
    
    def set_sentence(self, sentence):
        with self.lock:
            self.sentence = sentence
    
    def get(self):
        with self.lock:
            return (self.detected_letter, self.detected_confidence, 
                    self.hold_progress, self.sentence)
    
    def set_hold_time(self, hold_time):
        with self.lock:
            self.hold_time = hold_time
    
    def get_hold_time(self):
        with self.lock:
            return self.hold_time
    
    def set_sentence_mode(self, mode):
        with self.lock:
            self.sentence_mode = mode
    
    def get_sentence_mode(self):
        with self.lock:
            return self.sentence_mode

# Create shared state instance
if "shared_state" not in st.session_state:
    st.session_state.shared_state = SharedState()

shared_state = st.session_state.shared_state


class ASLVideoProcessor(VideoProcessorBase):
    """Video processor for live ASL detection with hold-time tracking."""
    
    def __init__(self):
        self.shared_state = None
        self.last_letter = None
        self.letter_start_time = None
        self.letter_appended = False
        self.frame_count = 0
        self.skip_frames = 3  # Process every 3rd frame for performance
        self.last_result_img = None
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if detector is None or classifier is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Lower resolution for faster processing (320x240 instead of 640x480)
        img = cv2.resize(img, (320, 240))
        
        # Skip frames for performance - return last result if skipping
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            if self.last_result_img is not None:
                return av.VideoFrame.from_ndarray(self.last_result_img, format="bgr24")
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Detect hands
        results = detector(img, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        CONFIDENCE_THRESHOLD = 0.5
        detected_letter = None
        detected_conf = 0.0
        
        for xyxy, mask, confidence, class_id, tracker_id, data in detections:
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            
            left, top, right, bottom = map(int, xyxy)
            if top < img.shape[0] * 0.3:
                continue
            
            hand_crop = img[top:bottom, left:right]
            
            try:
                clf_results = classifier(hand_crop, verbose=False)[0]
                clf_probs = clf_results.probs
                
                class_name = classifier.names[clf_probs.top1]
                class_conf = clf_probs.top1conf.item()
                
                detected_letter = class_name
                detected_conf = class_conf
                
                # Draw bounding box and label
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img, f"{class_name} ({class_conf:.2f})", 
                           (left, max(top - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
            except:
                pass
        
        # Handle hold-time logic for sentence builder
        hold_progress = 0.0
        current_time = time.time()
        
        if self.shared_state:
            hold_time = self.shared_state.get_hold_time()
            sentence_mode = self.shared_state.get_sentence_mode()
            
            if detected_letter:
                # Draw large letter in corner (scaled for 320x240)
                cv2.putText(img, detected_letter, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                if sentence_mode:
                    # Same letter being held
                    if detected_letter == self.last_letter:
                        if self.letter_start_time:
                            elapsed = current_time - self.letter_start_time
                            hold_progress = min(elapsed / hold_time, 1.0)
                            
                            # Draw progress bar on video (scaled for 320x240)
                            bar_width = int(150 * hold_progress)
                            cv2.rectangle(img, (85, 30), (235, 45), (100, 100, 100), -1)
                            cv2.rectangle(img, (85, 30), (85 + bar_width, 45), (0, 255, 0), -1)
                            cv2.putText(img, f"{elapsed:.1f}s/{hold_time:.0f}s", 
                                       (85, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            
                            # Append letter when hold time reached
                            if elapsed >= hold_time and not self.letter_appended:
                                self.shared_state.append_to_sentence(detected_letter)
                                self.letter_appended = True
                                # Visual feedback
                                cv2.putText(img, "ADDED!", (250, 40), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        # New letter - reset timer
                        self.last_letter = detected_letter
                        self.letter_start_time = current_time
                        self.letter_appended = False
                        hold_progress = 0.0
            else:
                # No detection - reset
                self.last_letter = None
                self.letter_start_time = None
                self.letter_appended = False
                hold_progress = 0.0
            
            # Update shared state
            self.shared_state.update_detection(detected_letter, detected_conf, hold_progress)
        
        # Save this frame as last result for frame skipping
        self.last_result_img = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def detect_from_image(image):
    """Detect and classify ASL letter from uploaded image."""
    if detector is None or classifier is None:
        return None, None, 0.0
    
    if image is None:
        return None, None, 0.0
    
    if isinstance(image, Image.Image):
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    frame = cv2.resize(frame, (640, 480))
    
    results = detector(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for uploaded images
    detected_letter = None
    detected_conf = 0.0
    
    # Debug: show number of detections
    num_detections = len(detections)
    
    for xyxy, mask, confidence, class_id, tracker_id, data in detections:
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        
        left, top, right, bottom = map(int, xyxy)
        # Removed top 30% filter for uploaded images - only applies to webcam
        
        hand_crop = frame[top:bottom, left:right]
        
        try:
            clf_results = classifier(hand_crop, verbose=False)[0]
            clf_probs = clf_results.probs
            
            class_name = classifier.names[clf_probs.top1]
            class_conf = clf_probs.top1conf.item()
            
            detected_letter = class_name
            detected_conf = class_conf
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({class_conf:.2f})", 
                       (left, max(top - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        except Exception as e:
            # Draw detection box even if classification fails
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
    result_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_image), detected_letter, detected_conf, num_detections


# Header
st.markdown("#ASL Detector")
st.markdown("American Sign Language Detection & Translation")

# Main layout
left_col, right_col = st.columns([1, 1], gap="large")

# Left Column: Input
with left_col:
    st.markdown("## Input Source")
    
    input_method = st.radio(
        "Choose input method:",
        ["Webcam (Live)", "Upload Image"],
        horizontal=True
    )
    
    if input_method == "Webcam (Live)":
        st.markdown("### Live Webcam Feed")
        st.caption("The detected letter and hold progress are shown directly on the video feed.")
        
        # Create video processor factory that passes shared state
        def video_processor_factory():
            processor = ASLVideoProcessor()
            processor.shared_state = shared_state
            return processor
        
        ctx = webrtc_streamer(
            key="asl-detector",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=video_processor_factory,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    else:
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Input Image", use_column_width=True)
            
            result_img, letter, confidence, num_detections = detect_from_image(input_image)
            
            st.caption(f"🔍 Found {num_detections} hand detection(s)")
            
            if result_img:
                st.image(result_img, caption="Detection Result", use_column_width=True)
            
            if letter:
                st.success(f"Detected: **{letter}** ({confidence:.1%})")
            elif num_detections == 0:
                st.warning("No hands detected in the image. Try an image with a clearer hand sign.")

# Right Column: Output
with right_col:
    st.markdown("## Detection Settings")
    
    detection_mode = st.radio(
        "Detection Mode:",
        ["Single Letter", "Sentence Builder"],
        horizontal=True
    )
    
    # Update shared state with mode
    shared_state.set_sentence_mode(detection_mode == "Sentence Builder")
    
    # Hold time slider for sentence builder
    if detection_mode == "Sentence Builder":
        hold_time = st.slider("Hold time to append (seconds)", 1.0, 5.0, 2.0, 0.5)
        shared_state.set_hold_time(hold_time)
        
        st.info(f"Hold the same sign for **{hold_time}** seconds to automatically append it to the sentence. Progress bar is shown on the video.")
    
    st.markdown("---")
    
    # Sentence Builder Controls
    if detection_mode == "Sentence Builder":
        # Auto-refresh to update sentence display (matches hold time)
        st_autorefresh(interval=int(hold_time * 1000), limit=None, key="sentence_refresh")
        
        st.markdown("### ✏️ Sentence Builder")
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⎵ Add Space", use_container_width=True):
                current = shared_state.get_sentence()
                shared_state.set_sentence(current + " ")
                st.rerun()
        
        with col2:
            if st.button("⌫ Backspace", use_container_width=True):
                current = shared_state.get_sentence()
                if current:
                    shared_state.set_sentence(current[:-1])
                    st.rerun()
        
        with col3:
            if st.button("🗑️ Clear All", use_container_width=True):
                shared_state.set_sentence("")
                st.rerun()
        
        # Display sentence
        st.markdown("### Constructed Sentence:")
        sentence = shared_state.get_sentence()
        sentence_text = sentence if sentence else "(empty - hold a sign to add letters)"
        st.markdown(f"## {sentence_text}")
        
        char_count = len(sentence)
        word_count = len(sentence.split()) if sentence.strip() else 0
        st.caption(f"📊 {char_count} characters | {word_count} words")
    
    else:
        st.markdown("### Single Letter Mode")
        st.info("👆 The detected letter is displayed directly on the video feed in the top-left corner.")

# ASL Reference
with st.expander("📖 ASL Reference Dictionary"):
    st.markdown("### American Sign Language Alphabet")
    st.image(
        "assets/asl_alphabet.png",
        caption="American Sign Language (ASL) Alphabet Reference",
        width=500
    )
