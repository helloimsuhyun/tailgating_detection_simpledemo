import cv2
import numpy as np
import pyrealsense2 as rs
from dt_apriltags import Detector


"""

apriltag를 사용해서 호모그라피 행렬을 찾는 것으로 결정. 찾아본 결과 aruco 마커보다 apriltage가 aruco마커보다 탐지에 rubust하다고 함

"""

# ----------------- RealSense  -----------------
pipeline = rs.pipeline()

def realsense_start():
    config = rs.config()        
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()

    return [ intr.fx , intr.fy, intr.ppx, intr.ppy ]        

def realsense_cam():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame : 
        return None
    
    color = np.asanyarray(color_frame.get_data()) 
    return color

def realsense_end():
    pipeline.stop()



# ----------------- apriltag -----------------
TAG_SIZE = 0.10 #m 단위

at_detector = Detector(
    searchpath=['apriltags'],
    families='tag36h11',
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

def draw_axes(vis_img, R, t, camera_intr, axis_len=0.05):
    """
    AprilTag pose(R, t)를 이용해 x,y,z 축을 화면에 그려준다.
    axis_len: 축 길이 (미터)
    camera_intr: [fx, fy, cx, cy]
    """

    fx, fy, cx, cy = camera_intr

    # tag 좌표계에서 축점들
    axes_points = np.array([
        [0, 0, 0],                 # origin
        [axis_len, 0, 0],          # x+
        [0, axis_len, 0],          # y+
        [0, 0, -axis_len]           # z+
    ]).T   # 3x4 형태로 (R @ P + t)를 위해 전치

    # 카메라 좌표계로 변환
    cam_points = R @ axes_points + t  # (3x3 @ 3x4 + 3x1)

    # 카메라 3D → 2D projection
    uv = []
    for i in range(cam_points.shape[1]):
        X, Y, Z = cam_points[:, i]
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        uv.append((int(u), int(v)))

    origin, x_axis, y_axis, z_axis = uv

    # 축 그리기
    cv2.line(vis_img, origin, x_axis, (0, 0, 255), 2)   # X축 (RED)
    cv2.line(vis_img, origin, y_axis, (0, 255, 0), 2)   # Y축 (GREEN)
    cv2.line(vis_img, origin, z_axis, (255, 0, 0), 2)   # Z축 (BLUE)

    return vis_img

def detect_apriltag():
    global camera_intr

    while True : 
        frame = realsense_cam()
        detections = None 
        if frame is None : 
            print("camera open failed ...")
            continue
        
        vis_frame = frame.copy()
        cv2.putText(
            vis_frame,
            "Press 'k' : detect tag   |   Press 'q' : quit",
            (10, 30),  # 화면 왼쪽 위 위치
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),  # 노란색
            2
        )

        cv2.imshow("realsense show ", vis_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('k') :
            detect_frame = frame.copy()
            vis_img = frame.copy()

            detect_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
            tags = at_detector.detect(
                detect_frame,
                estimate_tag_pose= True,
                camera_params = camera_intr,
                tag_size=TAG_SIZE
            )
            detections = tags
 

            for tag in tags:
                corners = tag.corners.astype(int)
                cx, cy = tag.center
                cx, cy = int(cx), int(cy)
                id_ = tag.tag_id
                R = tag.pose_R  # 3x3
                t = tag.pose_t  # 3x1

                # 감지 태그 시각화
                for idx in range(len(tag.corners)):
                    cv2.line(vis_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))                
                cv2.circle(vis_img, (cx, cy), 4, (0, 0, 255), -1)

                cv2.putText(
                    vis_img,
                    f"ID:{id_}",
                    (corners[0][0], corners[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                vis_img = draw_axes(vis_img, R, t, camera_intr, axis_len=TAG_SIZE * 0.7)

            cv2.imshow("detect of teg",vis_img)
        
        if key == ord('q'):
            return detections
        

            
        

if __name__ == "__main__":
    camera_intr = realsense_start()
    frame = realsense_cam()
    
    try:
        detect_apriltag()
        
    finally :
        realsense_end()
        cv2.destroyAllWindows()



