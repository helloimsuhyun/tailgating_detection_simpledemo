from ultralytics import YOLO
import pyrealsense2 as rs
import cv2
import os
import numpy as np
from datetime import datetime

# ----------------- 황일겸과 통신  -----------------

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import asyncio
import cv2
import numpy as np
import base64
import requests

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
#   WebSocket Frame Queue
# ----------------------
frame_queue = asyncio.Queue(maxsize=1)   # 최신 프레임만 유지


async def send_frames(websocket: WebSocket):
    """비동기적으로 queue에서 프레임 읽어 WebSocket으로 전송"""
    while True:
        jpg_bytes = await frame_queue.get()
        await websocket.send_bytes(jpg_bytes)


@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    await websocket.accept()
    print("📡 Client connected")

    try:
        await send_frames(websocket)
    except Exception as e:
        print("⚠️ WebSocket disconnected:", e)
    finally:
        await websocket.close()


def update_stream_frame(frame):
    """YOLO 루프에서 호출 — JPEG 변환 → queue에 넣음"""

    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return
    jpg_bytes = buffer.tobytes()

    # queue 최신화 (이전 프레임 버림)
    try:
        while not frame_queue.empty():
            frame_queue.get_nowait()
    except:
        pass

    # 최신 프레임 저장
    try:
        frame_queue.put_nowait(jpg_bytes)
    except asyncio.QueueFull:
        pass


def start_stream_server(host="0.0.0.0", port=5000):
    def _run():
        uvicorn.run(app, host=host, port=port, log_level="info")

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    print(f"[STREAM] WebSocket server running ws://{host}:{port}/ws/video")


def frame_to_base64(frame):
    ret, buf = cv2.imencode('.jpg', frame)   # numpy → jpeg bytes
    if not ret:
        return None
    return base64.b64encode(buf).decode()    # bytes → base64 문자열

SERVER_URL = "http://172.17.75.184:8000/door_event"

def send_door_event(log):
    try:
        r = requests.post(SERVER_URL, json=log, timeout=0.5)  # 타임아웃 짧게
        # 필요하면 응답 체크
        # print(r.status_code, r.text)
    except Exception as e:
        print("[ERROR] send_door_event:", e)


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
            "frame": frame_to_base64(frame)
        }

        cv2.imshow("Door Event", frame)
        cv2.waitKey(1)

        door_log.append(log)
        print(f"[DOOR] ID:{log['id']} | Time:{log['timestamp']} | Event:{log['event']}")

        send_door_event(log)


# roi 변수(초기값)
ROI_OUTER = None
ROI_INNER = None
ROI_DONT_CARE = None



#zone_table
ZONE_ORDER = {
    "OUTSIDE": 0,    # 밖
    "OUTER_ROI": 1,  # 안2
    "INNER_ROI": 2,  # 안1
}


frame_idx = 0
FRAME_LOST_THRESH = 30 #이거 이상 안보이면 사라진 것으로 처리

# ----------------- judge zone function -----------------
def get_zone(cx,cy,roi_in, roi_out ,roi_dontcare):

    in_inner = cv2.pointPolygonTest(roi_in.astype(np.float32),
                                (float(cx), float(cy)),
                                False) >= 0

    # OUTER 안에 있는지
    in_outer = cv2.pointPolygonTest(roi_out.astype(np.float32),
                                    (float(cx), float(cy)),
                                    False) >= 0
    
    if roi_dontcare is not None : 
        in_doncare = cv2.pointPolygonTest(roi_dontcare.astype(np.float32),
                                        (float(cx), float(cy)),
                                        False) >= 0
        if in_inner:
            zone = "INNER_ROI"
        elif in_doncare:
            return None
        elif in_outer :
            zone = "OUTER_ROI"
        else:
            zone = "OUTSIDE"

    else :
        if in_inner:
            zone = "INNER_ROI"
        elif in_outer:
            zone = "OUTER_ROI"
        else:
            zone = "OUTSIDE"

    return zone

# ----------------- state machine -----------------

def update_state(result,roi_in,roi_out,roi_dontcare,frame):
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
        
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = y2
        #cy = (y1 + y2) / 2.0

        current_zone = get_zone(cx,cy,roi_in,roi_out,roi_dontcare)

        if current_zone is None: #roi_doncare에 해당되면 본 것으로 처리하지 않음 그냥 안본 것으로 처리
            continue

        seen_id.append(id_)
            
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

                print(f"[LOST] id={id_}, first={first_zone}, last_zone={last_zone}, last_state={last_state}, final={final_state}")
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
    첫 프레임에서 INNER_ROI, OUTER_ROI, DONT_CARE_ROI를
    마우스로 4점씩 선택
    순서: INNER(안) -> OUTER(밖) -> DONT_CARE
    왼쪽 클릭: 점 선택, r: 리셋, Enter/Space: 확정
    """
    global ROI_OUTER, ROI_INNER, ROI_DONT_CARE, _current_roi_name, _tmp_points

    cv2.namedWindow("Set ROI")
    cv2.setMouseCallback("Set ROI", _mouse_callback)

    frame_for_draw = first_frame.copy()

    # 이미 확정된 ROI들을 항상 같이 그려주는 헬퍼
    def draw_fixed_rois(img):
        # INNER: 초록
        if ROI_INNER is not None and len(ROI_INNER) == 4:
            cv2.polylines(img, [ROI_INNER.astype(np.int32)], True, (0, 255, 0), 2)
        # OUTER: 파랑
        if ROI_OUTER is not None and len(ROI_OUTER) == 4:
            cv2.polylines(img, [ROI_OUTER.astype(np.int32)], True, (255, 0, 0), 2)
        # DONT_CARE: 노랑
        if ROI_DONT_CARE is not None and len(ROI_DONT_CARE) == 4:
            cv2.polylines(img, [ROI_DONT_CARE.astype(np.int32)], True, (0, 255, 255), 2)

    # ---------------- 1) INNER_ROI 선택 (안) ----------------
    _current_roi_name = "INNER"
    _tmp_points = []

    while True:
        disp = frame_for_draw.copy()
        draw_fixed_rois(disp)

        cv2.putText(disp, "INNER ROI (inside): 4 points, r=reset, ENTER=done",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 현재 찍는 점들
        for p in _tmp_points:
            cv2.circle(disp, p, 4, (0, 255, 0), -1)
        if len(_tmp_points) >= 2:
            cv2.polylines(disp, [np.array(_tmp_points, np.int32)], False, (0, 255, 0), 2)

        cv2.imshow("Set ROI", disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            _tmp_points = []
        elif key in [13, 32]:  # ENTER or SPACE
            if len(_tmp_points) == 4:
                ROI_INNER = np.array(_tmp_points, dtype=np.float32)
                print("INNER_ROI set to:", ROI_INNER)
                break

    # ---------------- 2) OUTER_ROI 선택 (밖) ----------------
    _current_roi_name = "OUTER"
    _tmp_points = []

    while True:
        disp = frame_for_draw.copy()
        draw_fixed_rois(disp)

        cv2.putText(disp, "OUTER ROI (outside): 4 points, r=reset, ENTER=done",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        for p in _tmp_points:
            cv2.circle(disp, p, 4, (255, 0, 0), -1)
        if len(_tmp_points) >= 2:
            cv2.polylines(disp, [np.array(_tmp_points, np.int32)], False, (255, 0, 0), 2)

        cv2.imshow("Set ROI", disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            _tmp_points = []
        elif key in [13, 32]:
            if len(_tmp_points) == 4:
                ROI_OUTER = np.array(_tmp_points, dtype=np.float32)
                print("OUTER_ROI set to:", ROI_OUTER)
                break

       # ---------------- 3) DONT_CARE_ROI 선택 ----------------
    _current_roi_name = "DONT_CARE"
    _tmp_points = []

    while True:
        disp = frame_for_draw.copy()
        draw_fixed_rois(disp)

        cv2.putText(
            disp,
            "DONT_CARE ROI: 4 points, r=reset, ENTER=done, n=skip",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for p in _tmp_points:
            cv2.circle(disp, p, 4, (0, 255, 255), -1)
        if len(_tmp_points) >= 2:
            cv2.polylines(disp, [np.array(_tmp_points, np.int32)], False, (0, 255, 255), 2)

        cv2.imshow("Set ROI", disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            _tmp_points = []

        elif key == ord('n'):   # 👉 DONT_CARE 안 쓰고 스킵
            ROI_DONT_CARE = None
            print("DONT_CARE_ROI skipped")
            break

        elif key in [13, 32]:   # ENTER or SPACE
            if len(_tmp_points) == 4:
                ROI_DONT_CARE = np.array(_tmp_points, dtype=np.float32)
                print("DONT_CARE_ROI set to:", ROI_DONT_CARE)
                break

    cv2.destroyWindow("Set ROI")




if __name__ == "__main__":
    realsense_start()
    frame = realsense_cam()

    if frame is not None : 
        update_stream_frame(frame)

    start_stream_server(host="0.0.0.0", port=5000)
    setup_rois(frame)

    try:
        while True:
            frame_idx += 1  # 프레임 인덱스 증가

            frame, result = tracker_step(model)
            if frame is None or result is None:
                continue

            annotated = result.plot()
            annotated = update_state(result, ROI_INNER, ROI_OUTER, ROI_DONT_CARE, annotated)

            # ROI 시각화 (선택)
            if ROI_OUTER is not None:
                cv2.polylines(annotated, [ROI_OUTER.astype(np.int32)], True, (255, 0, 0), 2)

            if ROI_INNER is not None:
                cv2.polylines(annotated, [ROI_INNER.astype(np.int32)], True, (0, 255, 0), 2)

            if ROI_DONT_CARE is not None:
                cv2.polylines(annotated, [ROI_DONT_CARE.astype(np.int32)], True, (0, 255, 255), 2) 

            update_stream_frame(annotated)
            cv2.imshow("RealSense YOLO Tracking Test", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        realsense_end()
        cv2.destroyAllWindows()

                          