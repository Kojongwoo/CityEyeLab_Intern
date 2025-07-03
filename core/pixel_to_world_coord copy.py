import cv2
import numpy as np

# 라벨 색상 및 이름 정의
LABEL_COLORS = {
    0: (0, 255, 0),      # Green
    1: (255, 0, 0),      # Blue
    2: (0, 0, 255),      # Red
    3: (0, 255, 255),    # Yellow
    4: (255, 0, 255),    # Magenta
    5: (255, 255, 0),    # Cyan
}
DEFAULT_COLOR = (200, 200, 200)
LABEL_NAMES = {
    0: 'car',
    1: 'bus_s',
    2: 'bus_m',
    3: 'truck_s',
    4: 'truck_m',
    5: 'truck_x',
    6: 'bike'
}

def draw_transformed_box(warped_img, M, x1, y1, x2, y2, obj_id, label):
    color = LABEL_COLORS.get(label, DEFAULT_COLOR)
    label_name = LABEL_NAMES.get(label, f"Label:{label}")
    
    # 바운딩 박스 꼭짓점 4개 좌표
    box_pts = np.float32([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ]).reshape(-1, 1, 2)
    
    # 투시 변환
    transformed_pts = cv2.perspectiveTransform(box_pts, M)
    transformed_pts = transformed_pts.reshape(-1, 2).astype(int)

    # 선 그리기
    for i in range(4):
        pt1 = tuple(transformed_pts[i])
        pt2 = tuple(transformed_pts[(i + 1) % 4])
        cv2.line(warped_img, pt1, pt2, color, 2)

    # 라벨 그리기
    label_pos = tuple(transformed_pts[0])
    cv2.putText(warped_img, f"ID:{obj_id}, {label_name}", label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# 좌표 클릭용
clicked_points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"Point {len(clicked_points)}: ({x}, {y})")

def select_points_from_image(image):
    global clicked_points
    clicked_points = []

    clone = image.copy()
    cv2.namedWindow("Select 4 Points")
    cv2.setMouseCallback("Select 4 Points", mouse_callback)

    while True:
        display = clone.copy()
        for i, pt in enumerate(clicked_points):
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
            cv2.putText(display, str(i+1), (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Select 4 Points", display)
        key = cv2.waitKey(1)
        if key == ord('q') or len(clicked_points) == 4:
            break

    cv2.destroyAllWindows()
    return np.float32(clicked_points)

# 바운딩 박스 라벨 데이터 불러오기
def read_raw_data(path):
    frame_data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = list(map(int, line.split(',')))
            frame, obj_id, x1, y1, x2, y2, label = values
            if frame not in frame_data:
                frame_data[frame] = []
            frame_data[frame].append((obj_id, x1, y1, x2, y2, label))
    return frame_data

# 전체 처리
def process_video_with_perspective(video_path, label_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] 비디오 열기 실패: {video_path}")
        return

    # 라벨 로드
    frame_data = read_raw_data(label_path)

    # 첫 프레임에서 사용자 클릭
    ret, first_frame = cap.read()
    if not ret:
        print("[오류] 첫 프레임을 읽을 수 없습니다.")
        return

    print("👉 영상에서 투시 변환할 4점을 클릭하세요 (좌하, 우하, 좌상, 우상 순으로)")
    src_pts = select_points_from_image(first_frame)

    # 출력 해상도 설정
    width, height = 800, 800
    dst_pts = np.float32([
        [0, height],
        [width, height],
        [0, 0],
        [width, 0]
    ])

    # 변환 행렬 계산
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 투시 변환 적용
        warped = cv2.warpPerspective(frame, M, (width, height))

        # 원본에 바운딩 박스 표시
        if frame_idx in frame_data:
            for obj_id, x1, y1, x2, y2, label in frame_data[frame_idx]:
                color = LABEL_COLORS.get(label, DEFAULT_COLOR)
                label_name = LABEL_NAMES.get(label, f"Label:{label}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{obj_id}, {label_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 보정된 영상에는 변환된 박스 그림
            draw_transformed_box(warped, M, x1, y1, x2, y2, obj_id, label)


        cv2.imshow("Original + Boxes", frame)
        cv2.imshow("Perspective View", warped)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    label_path = './assets/2024-10-21 08_56_19.337.txt'
    video_path = './assets/2024-10-21 08_56_19.337.mp4'
    process_video_with_perspective(video_path, label_path)
