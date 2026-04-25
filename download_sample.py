import urllib.request
import os

url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
filepath = "crowd.mp4"

print(f"Downloading sample crowd video from {url}...")
try:
    urllib.request.urlretrieve(url, filepath)
    print(f"Successfully downloaded to {filepath}")
except Exception as e:
    print(f"Failed to download video: {e}")
