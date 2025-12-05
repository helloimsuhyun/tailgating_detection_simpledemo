from ultralytics import YOLO
import pyrealsense2 as rs
import cv2
import os
import numpy as np
from datetime import datetime

# ----------------- 경로 설정 -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(BASE_DIR, "track_yaml", "botsort.yaml")

# ----------------- RealSense / YOLO -----------------
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


# ----------------- tracker -----------------
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

# ----------------- roi / id state global -----------------

# id - state / last frame / first_zone / current_zone
track_states = {} # ENTER , ENTER_DEEP ,IN, OUT, LOST , NONE
track_last_seen = {}
track_state_change_time = {} 
track_snap_shot = {} # 상태변이때의 snapshot



track_first_zone = {} #처음 id가 보인 zone
track_current_zone = {} #현재 zone
track_prev_zone = {} #이전 frame에서의 zone

door_log = []

def add_door_log(id, final_state, frame):

    if final_state == "DOOR_EXIT" : 

        log = {
            "id": id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": final_state,
            "frame": frame.copy()
        }

        print(f"[DOOR] ID:{log['id']} | Time:{log['timestamp']} | Event:{log['event']}")
        cv2.imshow("Door Event", log["frame"])
        cv2.waitKey(1)

        door_log.append(log)


# roi 변수(초기값)
ROI_OUTER = np.array([
    [379, 163],  # 왼쪽 위
    [442, 155],  # 오른쪽 위
    [430, 362],  # 왼쪽 아래
    [370, 334],  # 오른쪽 아래
], dtype=np.float32)

ROI_INNER = np.array([
    [379, 163],  # 왼쪽 위
    [442, 155],  # 오른쪽 위
    [430, 362],  # 왼쪽 아래
    [370, 334],  # 오른쪽 아래
], dtype=np.float32)


#zone_table
ZONE_ORDER = {
    "OUTSIDE": 0,    # 밖
    "OUTER_ROI": 1,  # 안2
    "INNER_ROI": 2,  # 안1
}


frame_idx = 0
FRAME_LOST_THRESH = 30 #이거 이상 안보이면 사라진 것으로 처리

# ----------------- judge zone function -----------------
def get_zone(cx,cy,roi_in, roi_out):

    in_inner = cv2.pointPolygonTest(roi_in.astype(np.float32),
                                (float(cx), float(cy)),
                                False) >= 0

    # OUTER 안에 있는지
    in_outer = cv2.pointPolygonTest(roi_out.astype(np.float32),
                                    (float(cx), float(cy)),
                                    False) >= 0

    if in_inner:
        zone = "INNER_ROI"
    elif in_outer:
        zone = "OUTER_ROI"
    else:
        zone = "OUTSIDE"

    return zone

# ----------------- state machine -----------------

def update_state(result,roi_in,roi_out,frame):
    global track_current_zone, track_last_seen, track_first_zone ,track_states , track_state_change_time, track_prev_zone
    global frame_idx

    boxes = result.boxes

    if boxes.id is None: #화면에 아무도 잡히지 않은 경우
        seen_id = []
        handle_lost_state(frame_idx, seen_id)
        return frame

    ids = boxes.id.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    seen_id = []

    for id_, box in zip(ids, xyxy): #현재 프레임에서 보이는 ID들에 대한 처리
        seen_id.append(id_)
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = y2
        #cy = (y1 + y2) / 2.0

        current_zone = get_zone(cx,cy,roi_in,roi_out)
        prev_zone = track_current_zone.get(id_, None)
        state = track_states.get(id_,None)

        if track_first_zone.get(id_,None) is None : # 처음 보인 ZONE이 정의되지 않음 -> 정의해줌
            track_first_zone[id_] = current_zone
        
        
        if prev_zone is not None and prev_zone != current_zone : #zone이 변화한 경우 상태전이
            #들어오는 방향
            if prev_zone == "OUTER_ROI" and current_zone == "INNER_ROI":
                state = "ENTER_DEEP"
            elif prev_zone == "OUTSIDE" and current_zone == "OUTER_ROI"  :
                state = "ENTER"
            
            #나가는 방향
            elif prev_zone == "INNER_ROI" and current_zone == "OUTER_ROI"  :
                state = "EXIT"
            elif prev_zone == "INNER_ROI" and current_zone == "OUTSIDE"  :
                state = "EXIT_DEEP"
            
            track_snap_shot[id_] = frame
            track_state_change_time[id_] = frame_idx
        else :
            last_change = track_state_change_time.get(id_, None)
            if (last_change is not None and frame_idx - last_change > 150) and current_zone == "OUTSIDE" :
                state = "NONE"

        #state, zone, last_seen frame idx 업데이트
        track_states[id_] = state
        track_current_zone[id_] = current_zone 
        track_prev_zone[id_] = current_zone
        track_last_seen[id_] = frame_idx

        label = f"ID {id_} | {current_zone} | {state}"
        cv2.putText(
            frame,
            label,
            (int(x1), int(y1) - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    
    handle_lost_state(frame_idx,seen_id)
    return frame

def handle_lost_state(frame_idx,seen_id):
    """
    frame_idx : 이번 프레임의 인덱스
    seen_id : 이번 프레임에서 보인 id들의 배열
    """
    global track_current_zone, track_last_seen, track_first_zone, track_states,track_prev_zone
    
    to_del = []

    for id_ in list(track_current_zone.keys()): #현재 추적중인 id들을 불러옴
        if id_ not in seen_id : #현재 프레임에서 안보이는 녀석들
            last_seen = track_last_seen.get(id_,None)

            if last_seen is not None and (frame_idx - last_seen > FRAME_LOST_THRESH): #이렇게 되면 LOST임
                last_state = track_states.get(id_,None)
                last_zone = track_prev_zone.get(id_,None)
                first_zone = track_first_zone.get(id_, None)

                final_state = decide_final_state(last_zone,last_state,first_zone)
                final_snap_shot = track_snap_shot.get(id_,None)
                add_door_log(id_,final_state,final_snap_shot)
                to_del.append(id_)
    
    for id_ in to_del :
        track_current_zone.pop(id_, None)
        track_prev_zone.pop(id_, None)
        track_last_seen.pop(id_, None)
        track_first_zone.pop(id_,None)
        track_states.pop(id_, None)
        track_state_change_time.pop(id_, None)

def decide_is_he_door_in(last_zone, last_state, first_zone):
    if (first_zone == "INNER_ROI" or first_zone == "OUTER_ROI") and last_zone == "OUTSIDE" and (last_state == "EXIT" or last_state == "EXIT_DEEP"): 
        return "DOOR IN"
    
def decide_final_state(last_zone, last_state, first_zone): # 프레임에서 사라진 id_들을 care, 1. 문 밖에서 안으로 IN 한 경우 2.CCTV 영역 밖으로 벗어난 경우
    
    if first_zone != "INNER_ROI" : #first zone 이 out outer roi 중에 하나면서
        if last_zone == "INNER_ROI" : 
            return "DOOR_EXIT"
        if last_zone == "OUTER_ROI" : 
            if last_state == "ENTER":
                return "DOOR_EXIT"   
        
    if last_zone == "OUTSIDE" and (last_state == None or last_state == "NONE"): 
        return "OUT OF CCTV RANGE"

    return "UNKNOWN"


# ----------------- set roi by mouse (by gpt) -----------------
# ----------------- ROI 마우스 선택 -----------------
_current_roi_name = None      # "OUTER" 또는 "INNER"
_tmp_points = []              # 지금 선택 중인 4점


def _mouse_callback(event, x, y, flags, param):
    global _tmp_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(_tmp_points) < 4:
            _tmp_points.append((x, y))
            print(f"[{_current_roi_name}] point {_tmp_points}")


def setup_rois(first_frame):
    """
    첫 프레임에서 OUTER_ROI, INNER_ROI를 마우스로 4점씩 선택
    왼쪽 클릭으로 점 선택, r로 리셋, Enter/Space로 확정
    """
    global ROI_OUTER, ROI_INNER, _current_roi_name, _tmp_points

    cv2.namedWindow("Set ROI")
    cv2.setMouseCallback("Set ROI", _mouse_callback)

    frame_for_draw = first_frame.copy()

    # 1) OUTER_ROI 선택
    _current_roi_name = "OUTER"
    _tmp_points = []

    while True:
        disp = frame_for_draw.copy()
        cv2.putText(disp, "OUTER ROI: 4 point, r=reset, ENTER = end",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

        # 이미 찍은 점들 그리기
        for p in _tmp_points:
            cv2.circle(disp, p, 4, (255, 0, 0), -1)
        if len(_tmp_points) >= 2:
            cv2.polylines(disp, [np.array(_tmp_points, np.int32)], False, (255,0,0), 2)

        cv2.imshow("Set ROI", disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):   # reset
            _tmp_points = []
        elif key in [13, 32]:  # ENTER 또는 SPACE
            if len(_tmp_points) == 4:
                ROI_OUTER = np.array(_tmp_points, dtype=np.float32)
                print("OUTER_ROI set to:", ROI_OUTER)
                break

    # 2) INNER_ROI 선택
    _current_roi_name = "INNER"
    _tmp_points = []

    while True:
        disp = frame_for_draw.copy()
        cv2.putText(disp, "INNER ROI: 4 point, r=reset, ENTER= end",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        for p in _tmp_points:
            cv2.circle(disp, p, 4, (0, 255, 0), -1)
        if len(_tmp_points) >= 2:
            cv2.polylines(disp, [np.array(_tmp_points, np.int32)], False, (0,255,0), 2)

        cv2.imshow("Set ROI", disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            _tmp_points = []
        elif key in [13, 32]:  # ENTER or SPACE
            if len(_tmp_points) == 4:
                ROI_INNER = np.array(_tmp_points, dtype=np.float32)
                print("INNER_ROI set to:", ROI_INNER)
                break

    cv2.destroyWindow("Set ROI")



if __name__ == "__main__":
    realsense_start()
    frame = realsense_cam()
    setup_rois(frame)

    try:
        while True:
            frame_idx += 1  # 프레임 인덱스 증가

            frame, result = tracker_step(model)
            if frame is None or result is None:
                continue

            annotated = result.plot()
            annotated = update_state(result, ROI_INNER, ROI_OUTER, annotated)

            # ROI 시각화 (선택)
            cv2.polylines(annotated, [ROI_OUTER.astype(np.int32)], True, (255, 0, 0), 2)
            cv2.polylines(annotated, [ROI_INNER.astype(np.int32)], True, (0, 255, 0), 2)

            cv2.imshow("RealSense YOLO Tracking Test", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        realsense_end()
        cv2.destroyAllWindows()

                          