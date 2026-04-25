import cv2
import numpy as np
import time

def create_dummy_video():
    print("Creating dummy test.mp4 video for demo...")
    width, height = 640, 480
    fps = 30
    duration = 5 # seconds
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test.mp4', fourcc, fps, (width, height))
    
    for i in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw some moving "people" so heatmap has something to show, 
        # though YOLOv8 might not identify them as 'person', 
        # it gives the application a video file to play instead of crashing.
        x1 = (i * 5) % width
        y1 = height // 2
        
        x2 = width - ((i * 3) % width)
        y2 = height // 3
        
        cv2.rectangle(frame, (x1, y1), (x1+50, y1+150), (200, 200, 200), -1)
        cv2.rectangle(frame, (x2, y2), (x2+60, y2+180), (150, 150, 250), -1)
        
        cv2.putText(frame, "CCTV DEMO", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
        
    out.release()
    print("test.mp4 created successfully.")

if __name__ == '__main__':
    create_dummy_video()
