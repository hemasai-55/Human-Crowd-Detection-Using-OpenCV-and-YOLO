from flask import Flask, render_template, Response, jsonify
import cv2
import time
import csv
import os
import threading
from detector import CrowdDetector
from crowd_logic import CrowdLogic

app = Flask(__name__)

# Settings
VIDEO_PATH = "crowd.mp4"

# Global data block
current_data = {
    "total_people": 0,
    "highest_density": 0,
    "zones": ["LOW"]*9,
    "zone_counts": [0]*9,
    "overall_level": "LOW",
    "action": "Normal Entry",
    "timestamp": ""
}
data_lock = threading.Lock()

# Initialize objects globally but lazily
detector = None
logic = None

latest_frame_bytes = None

def init_system():
    global detector, logic
    if detector is None:
        detector = CrowdDetector()
        logic = CrowdLogic(frame_width=640, frame_height=480)

def detection_loop():
    global current_data, latest_frame_bytes
    init_system()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error: Video file not found or cannot be opened")
        os._exit(1)
    
    # Time tracker for logging
    last_log_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
                
        # Resize safely
        try:
            frame = cv2.resize(frame, (640, 480))
        except Exception as e:
            continue
            
        # Detect
        detections, _ = detector.detect(frame)
        
        # Process Logic
        processed_frame, data = logic.process(frame, detections)
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        data['timestamp'] = timestamp
        
        # Update global data
        with data_lock:
            current_data = data
            
        # Logging (write every 1 second)
        current_time = time.time()
        if current_time - last_log_time >= 1.0:
            log_data(timestamp, data['total_people'], data['overall_level'])
            last_log_time = current_time

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if ret:
            latest_frame_bytes = buffer.tobytes()

def generate_frames():
    global latest_frame_bytes
    while True:
        if latest_frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_bytes + b'\r\n')
        time.sleep(0.05)

def log_data(ts, count, level):
    file_exists = os.path.isfile('logs.csv')
    with open('logs.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "count", "level"])
        writer.writerow([ts, count, level])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def api_data():
    with data_lock:
        return jsonify(current_data)

if __name__ == '__main__':
    # Start detection looping in a background daemon thread
    threading.Thread(target=detection_loop, daemon=True).start()
    # Run the flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
