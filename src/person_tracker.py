from ultralytics import YOLO
import pyrealsense2 as rs
import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAM_PATH = os.path.join(BASE_DIR, "data", "ja-ma", "9_12", "video",
                        "2021-08-01_09-00-00_sun_sunny_out_ja-ma_C0041.mp4")
YAML_PATH = os.path.join(BASE_DIR, "track_yaml", "botsort.yaml")

pipeline = rs.pipeline()
model = YOLO("yolo11s.pt")  


def realsense_start():
    config = rs.config()        
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)        

def realsense_cam():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame : 
        return None
    
    color = np.asanyarray(color_frame.get_data()) 
    return color

def realsense_end():
    pipeline.stop()


def tracker_step(model):
    frame = realsense_cam()
    if frame is None:
        return None, None

    results = model.track(
        source=frame,
        tracker=YAML_PATH,
        conf=0.65,
        iou=0.5,
        classes=[0],
        persist=True, #track state 유지
        stream=False,    
        verbose=False, #log 제외
    )
    result = results[0]
    return frame, result


if __name__ == "__main__":
    realsense_start()

    try:
        while True:
            frame, result = tracker_step(model)
            if frame is None:
                continue

            annotated = result.plot()
            cv2.imshow("RealSense YOLO Tracking Test", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        realsense_end()
        cv2.destroyAllWindows()
                          