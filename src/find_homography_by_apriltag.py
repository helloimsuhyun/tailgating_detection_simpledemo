import cv2
import numpy as np
import pyrealsense2 as rs
from dt_apriltags import Detector


"""

apriltag를 사용해서 호모그라피 행렬을 찾는 것으로 결정. 찾아본 결과 aruco 마커보다 apriltage가 aruco마커보다 탐지에 rubust하다고 함
파이썬 바인딩을 사용 pip install dt-apriltags


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
TAG_SIZE = 0.065 #m 단위

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

    # tag local 좌표계 축점, 각 colums들이 좌표
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

    # 시각화
    cv2.line(vis_img, origin, x_axis, (0, 0, 255), 2)   # X축 (R)
    cv2.line(vis_img, origin, y_axis, (0, 255, 0), 2)   # Y축 (GR)
    cv2.line(vis_img, origin, z_axis, (255, 0, 0), 2)   # Z축 (B)

    return vis_img

def detect_apriltag():
    global camera_intr

    detection = None 

    while True : 
        frame = realsense_cam()
        if frame is None : 
            print("camera open failed ...")
            continue
        
        vis_frame = frame.copy()
        cv2.putText(
            vis_frame,
            "Press 'k' : detect tag   |   Press 'q' : quit",
            (10, 30),  
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255), 
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
            detection = tags
 
            for tag in tags:
                corners = tag.corners.astype(int)
                cx, cy = tag.center
                cx, cy = int(cx), int(cy)
                id_ = tag.tag_id
                R = tag.pose_R  # 3x3
                t = tag.pose_t  # 3x1

                #감지된 태그 시각화
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
            cv2.destroyAllWindows()
            return detection
        
def make_homograhpy(R,t):
    global camera_intr
    fx, fy, cx, cy = camera_intr
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])

    # s[u v 1] #image cordinate = K [R1 R2 R3 t] [ X Y Z 1] 인데 Z가 0 

    Rt_plane = np.column_stack([R[:, 0:2], t.reshape(3, 1)])  # 3x3 (X,Y,1)
    H = K @ Rt_plane

    #정규화
    H = H / H[2, 2]
    return H

def world_to_img(X,Y,H):
    XY1 = np.array([X, Y, 1.0])
    uv1 = H @ XY1
    u = uv1[0] / uv1[2]
    v = uv1[1] / uv1[2]
    return int(u), int(v)

def run_world_input_visualization(H):
    """
    H: 평면(X,Y,1) -> 이미지(u,v,1) homography

    동작:
      - 평소에는 실시간 스트림
      - 'e' 누르면 터미널에서 X Y (m) 입력 → 그 좌표에 점 + 좌표 텍스트를 계속 오버레이
      - 다시 'e' 누르고 다른 X Y 입력하면 점 위치 업데이트
      - 'q' 누르면 종료
    """
    print("[world->image 시각화 모드]")
    print("  - 실시간 스트림 보다가 'e' 누르면 X Y (m) 입력 모드")
    print("  - 예: 0.1 0.0")
    print("  - 'q' : 종료")

    current_coord = None  # (X, Y) in meters

    while True:
        frame = realsense_cam()
        if frame is None:
            print("camera frame is None")
            continue

        vis = frame.copy()

        # 현재 좌표가 있으면 점 + 텍스트 오버레이
        if current_coord is not None:
            X, Y = current_coord
            u, v = world_to_img(X, Y, H)

            # 점
            cv2.circle(vis, (u, v), 6, (0, 0, 255), -1)

            # 텍스트
            text = f"{X:.3f}, {Y:.3f} m"
            cv2.putText(
                vis,
                text,
                (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            vis,
            "Press 'e' to set (X,Y) in meters, 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow("world_to_image", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("quit visualization")
            break

        if key == ord('e'):
            # --- 여기서만 잠깐 터미널 입력 모드로 진입 ---
            raw = input("X Y (meter 단위, 예: 0.1 0.0) 입력 (취소: 엔터만): ").strip()
            if raw == "":
                print("입력 취소")
                continue

            try:
                xs, ys = raw.split()
                X = float(xs)
                Y = float(ys)
                current_coord = (X, Y)
                print(f"[SET] world=({X:.3f}, {Y:.3f}) m")
            except Exception as e:
                print("입력 형식 오류. 예: 0.1 0.0   (에러:", e, ")")
                continue


if __name__ == "__main__":
    camera_intr = realsense_start()
    frame = realsense_cam()
    
    try:
        dt = detect_apriltag()
        if dt is not None : print(dt[0].pose_R)
        
        if dt is not None and len(dt) == 1:
            tag = dt[0]
            R = tag.pose_R
            t = tag.pose_t
            H = make_homograhpy(R,t)

            run_world_input_visualization(H)
        
    finally :
        realsense_end()
        cv2.destroyAllWindows()



