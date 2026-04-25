import cv2
import os
from ultralytics import YOLO

class CrowdDetector:
    def __init__(self):
        # Load YOLOv8 nano model
        self.model = YOLO('yolov8n.pt')
        self.model.fuse() # Optimize model for inference
        
        # Load face cascade for privacy feature
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        # Run inference
        results = self.model(frame, classes=[0], verbose=False) # class 0 is person
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                if conf > 0.3:
                    detections.append((x1, y1, x2, y2))
                    
                    # Privacy feature: Face blurring
                    # Blurring the top 20% of bounding box as a simple and robust privacy feature
                    head_h = int((y2 - y1) * 0.2)
                    if head_h > 0 and y1+head_h < frame.shape[0]:
                        head_roi = frame[y1:y1+head_h, x1:x2]
                        if head_roi.shape[0] > 0 and head_roi.shape[1] > 0:
                            # Apply strong gaussian blur
                            head_roi = cv2.GaussianBlur(head_roi, (51, 51), 30)
                            frame[y1:y1+head_h, x1:x2] = head_roi
                            
        return detections, frame
