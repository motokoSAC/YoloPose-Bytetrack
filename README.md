# Intro
This is a simple implementation of YOLO-based real-time pose estimation integrated with ByteTrack, achieving over 30 FPS on an RTX 2070 GPU. The project uses a persistent ID tracker to perform ReID (Re-Identification) on the non-persistent ID results from YOLO pose estimation. However, the ReID performance depends on the tracker's capabilities. During testing, it was found that ByteTrack may lose track of IDs in cases of occlusion or when users leave the field of view. To address this, consider using a better tracker or fine-tuning the parameters to improve performance.
# Start
* pip install opencv-python, ultralytics, numpy
* cd to the directory
* python yolopose.py
