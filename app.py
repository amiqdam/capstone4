
import cv2
from PIL import Image
import supervision as sv
from ultralytics import YOLO

# Load best models
detector = YOLO("runs/detect/model_2/weights/best.pt")
classifier = YOLO("runs/classify/cls_model_3/weights/best.pt")

# Activate the webcamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open the webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))
    
    # Detect hand signs 
    results = detector(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Draw detections
    CONFIDENCE_THRESHOLD = 0.8
    for xyxy, mask, confidence, class_id, tracker_id, data in detections:
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        left, top, right, bottom = map(int, xyxy)
        if top < frame.shape[0] * 0.1:
            continue

        # Crop hand region
        hand_crop = frame[top:bottom, left:right]
        
        # Classify the cropped hand
        try:
            clf_results = classifier(hand_crop, verbose=False)[0]
            clf_probs = clf_results.probs
            
            # Get classification result
            class_name = classifier.names[clf_probs.top1]
            class_conf = clf_probs.top1conf.item()
            
            # Create label with classification
            label = f"{class_name} ({class_conf:.2f})"
        except:
            label = "Unknown"

        # Bounding box
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Label with classification
        cv2.putText(
            frame,
            label,
            (left, max(top - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Show result
    cv2.imshow("Hand Detection & Classification", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()