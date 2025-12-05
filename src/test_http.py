# http_test_sender.py
import requests
from datetime import datetime

SERVER_URL = "http://172.17.68.34:8000/door_event"  # 친구 IP

def main():
    log = {
        "id": 123,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": "TEST_EVENT",
        "frame": None,  # 일단 테스트라서 이미지 없이 보냄
    }

    try:
        r = requests.post(SERVER_URL, json=log, timeout=2)
        print("[SENDER] status:", r.status_code)
        print("[SENDER] response:", r.text)
    except Exception as e:
        print("[SENDER ERROR]", e)

if __name__ == "__main__":
    main()
