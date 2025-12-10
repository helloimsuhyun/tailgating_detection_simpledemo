import cv2
import numpy as np
import pyrealsense2 as rs
import os
from dt_apriltags import Detector
from ultralytics import YOLO


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
TAG_SIZE = 0.1 #m 단위 

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

                vis_img = draw_axes(vis_img, R, t, camera_intr, axis_len=TAG_SIZE * 0.5)

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

def img_to_world(u, v, H): # 역호모그래피
    uv1 = np.array([u, v, 1.0])
    H_inv = np.linalg.inv(H)       
    XY1 = H_inv @ uv1

    X = XY1[0] / XY1[2]
    Y = XY1[1] / XY1[2]
    return X, Y  

# -----------  yolo
model = YOLO("yolo11s.pt")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(BASE_DIR, "track_yaml", "botsort.yaml")

def tracker_step(model):
    frame = realsense_cam()
    if frame is None:
        return None, None

    results = model.track(
        source=frame,
        tracker=YAML_PATH,
        conf=0.55,
        iou=0.5,
        classes=[64],
        persist=True, #track state 유지
        stream=False,    
        verbose=False, #log 제외
    )
    result = results[0]
    return frame, result

track_xy = {} 

def update_track(result,frame):
    global track_xy

    if result is None or result.boxes is None:
        track_xy = {}
        return frame
    
    boxes = result.boxes

    if boxes.id is None: #화면에 아무도 잡히지 않은 경우
        track_xy = {}
        return frame

    ids = boxes.id.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    current_tracks = {}

    for id_, box in zip(ids, xyxy): #현재 프레임에서 보이는 ID들에 대한 처리
        
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = y2
        current_tracks[id_] = (cx, cy)
    track_xy = current_tracks

    return frame

# -----------  vis by gpt
MAP_SIZE = 600             # 맵 그림 사이즈 (픽셀)
MAP_SCALE = 200.0          # 1m당 몇 픽셀로 볼지 (원하는대로 조절)
MAP_ORIGIN = (MAP_SIZE//2, MAP_SIZE//2)  # (0,0) world를 맵 중앙에 둘 거임

def world_to_map_pixel(X, Y):
    """
    world 좌표계 (X,Y)를 2D 맵 이미지(px) 좌표로 변환
    맵 중앙이 (0,0) world, X 오른쪽 +, Y 위쪽 + 로 가정
    """
    ox, oy = MAP_ORIGIN

    # X: 오른쪽이 +, Y: 위쪽이 +가 되도록 (이미지 y는 아래가 +라서 -Y)
    mx = int(ox + X * MAP_SCALE)
    my = int(oy - Y * MAP_SCALE)

    return mx, my

def draw_grid(map_img, grid_step_m=0.005):
    """
    map_img    : 2D 맵 이미지 (H x W x 3)
    grid_step_m: 격자 간격 (미터 단위, 예: 0.5m, 1.0m)
    """
    h, w = map_img.shape[:2]
    ox, oy = MAP_ORIGIN

    # 1칸(격자)당 픽셀 수
    step_px = int(MAP_SCALE * grid_step_m)
    if step_px <= 0:
        return map_img

    # 연한 회색 격자
    grid_color = (220, 220, 220)

    # --- 수직선 (X 방향 격자) ---
    # origin에서 오른쪽으로
    x = ox
    while x < w:
        cv2.line(map_img, (x, 0), (x, h - 1), grid_color, 1)
        x += step_px

    # origin에서 왼쪽으로
    x = ox
    while x >= 0:
        cv2.line(map_img, (x, 0), (x, h - 1), grid_color, 1)
        x -= step_px

    # --- 수평선 (Y 방향 격자) ---
    # origin에서 아래쪽으로
    y = oy
    while y < h:
        cv2.line(map_img, (0, y), (w - 1, y), grid_color, 1)
        y += step_px

    # origin에서 위쪽으로
    y = oy
    while y >= 0:
        cv2.line(map_img, (0, y), (w - 1, y), grid_color, 1)
        y -= step_px

    # 중심 축은 조금 더 진하게 (X/Y axis)
    cv2.line(map_img, (ox, 0), (ox, h - 1), (180, 180, 180), 2)  # Y축
    cv2.line(map_img, (0, oy), (w - 1, oy), (180, 180, 180), 2)  # X축

    return map_img


def draw_2d_map(track_xy, H):
    """
    track_xy : {id: (u,v)}  이미지 좌표 (bottom center)
    H       : world ↔ image 호모그래피
    return  : 2D 맵 (np.ndarray)
    """
    # 흰색 바탕 맵 생성
    map_img = np.ones((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8) * 255

    # 🔹 격자 먼저 그리기
    map_img = draw_grid(map_img, grid_step_m=1.0)  # 1m 간격 (원하면 0.5로 줄여도 됨)

    # 중심점(0,0) 표시
    cv2.circle(map_img, MAP_ORIGIN, 4, (0, 0, 0), -1)
    cv2.putText(
        map_img,
        "(0,0)",
        (MAP_ORIGIN[0] + 5, MAP_ORIGIN[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 0),
        1
    )

    for id_, (u, v) in track_xy.items():
        # 1) 이미지 좌표 → world 좌표 (m)
        X, Y = img_to_world(u, v, H)

        # 2) world 좌표 → 맵 픽셀 좌표
        mx, my = world_to_map_pixel(X, Y)

        if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
            # 사람 위치 점
            cv2.circle(map_img, (mx, my), 6, (0, 0, 255), -1)

            # ID 텍스트
            cv2.putText(
                map_img,
                f"ID:{id_}",
                (mx + 8, my - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (50, 50, 50),
                1
            )

            # 월드 좌표 텍스트
            cv2.putText(
                map_img,
                f"({X:.2f}, {Y:.2f})",
                (mx + 8, my + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 0, 0),
                1
            )

    return map_img



if __name__ == "__main__":
    # 1) 카메라 시작
    camera_intr = realsense_start()

    H = None  

    try:
        dt = detect_apriltag()   

        if dt is None or len(dt) == 0:
            print("tag fail")
        else:
            tag = dt[0]
            R = tag.pose_R    # 3x3
            t = tag.pose_t    # 3x1
            H = make_homograhpy(R, t)
            print("Homography H:\n", H)

            while True:
                frame, result = tracker_step(model)
                annotated = result.plot()
                if frame is None or result is None:
                    continue

                frame_vis = update_track(result, frame)
                cv2.imshow("Tracking", annotated)

                if H is not None and len(track_xy) > 0:
                    map_img = draw_2d_map(track_xy, H)
                    cv2.imshow("BEV Map", map_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    finally:
        realsense_end()
        cv2.destroyAllWindows()




