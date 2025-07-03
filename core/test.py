import cv2
import numpy as np

# 차선 정의: 여러 개
lines = {
    "line1": ((200, 400), (800, 400)),
    "line2": ((200, 500), (800, 500)),
}

# 트래킹 객체의 이전 / 현재 중심 좌표
# 예시: object_id: ((x_prev, y_prev), (x_curr, y_curr))
tracked_objects = {
    1: ((300, 390), (300, 410)),  # line1 통과
    2: ((400, 490), (400, 510)),  # line2 통과
    3: ((100, 300), (100, 310)),  # 통과 아님
}

# 객체별로 이미 어떤 선을 통과했는지 기록 (중복 방지용)
triggered_lines = {}

# ──────────────────────────────────────
# 유틸 함수: 선분 교차 여부 판단
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def do_lines_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# ──────────────────────────────────────
# 이벤트 처리 함수
def handle_line_crossing(object_id, line_name):
    print(f"[EVENT] 객체 {object_id} 가 {line_name} 을 통과했습니다.")
    # 여기에 DB, 로그, 알림 등의 처리를 추가할 수 있음

# ──────────────────────────────────────
# 화면 초기화 (시각화용)
frame = np.zeros((600, 1000, 3), dtype=np.uint8)

# 차선 시각화
for line_name, (start, end) in lines.items():
    cv2.line(frame, start, end, (0, 0, 255), 2)
    cv2.putText(frame, line_name, start, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# ──────────────────────────────────────
# 트래킹된 객체 순회
for obj_id, (prev, curr) in tracked_objects.items():
    if obj_id not in triggered_lines:
        triggered_lines[obj_id] = set()

    # 점 표시
    cv2.circle(frame, prev, 5, (255, 0, 0), -1)
    cv2.circle(frame, curr, 5, (0, 255, 0), -1)
    cv2.line(frame, prev, curr, (255, 255, 0), 1)

    for line_name, (start, end) in lines.items():
        if line_name in triggered_lines[obj_id]:
            continue  # 이미 통과한 선은 스킵

        if do_lines_intersect(prev, curr, start, end):
            handle_line_crossing(obj_id, line_name)
            triggered_lines[obj_id].add(line_name)

# ──────────────────────────────────────
# 시각화
cv2.imshow("Tracking Line Crossing", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
