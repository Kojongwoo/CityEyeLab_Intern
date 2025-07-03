import numpy as np
import cv2

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

prev_positions = {}

# --- 마우스 드래그로 2개 영역 그리기용 전역 변수 ---
draw_points = []
drawing_done = False
drawing_window_name = "Draw 2 Areas: Click 4 points for area1, then 4 for area2"

def point_in_polygon(pt, polygon):
    return cv2.pointPolygonTest(polygon, pt, False) >= 0

def pixel_to_gps(x, y):
    gps1 = (37.401383, 127.112679)
    gps2 = (37.401371, 127.113207)
    px1 = (97, 415)
    px2 = (1342, 452)

    ratio = (x - px1[0]) / (px2[0] - px1[0])
    lat = gps1[0] + ratio * (gps2[0] - gps1[0])
    lon = gps1[1] + ratio * (gps2[1] - gps1[1])
    return (lat, lon)

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


def draw_polygon_with_mouse(event, x, y, flags, param):
    global draw_points, drawing_done
    if drawing_done:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(draw_points) < 8:
            draw_points.append((x, y))
            print(f"Point {len(draw_points)}: {(x,y)}")
            if len(draw_points) == 8:
                drawing_done = True
                print("영역 설정 완료!")

def get_area_polygons_from_user(video_path):
    global draw_points, drawing_done
    draw_points = []
    drawing_done = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] 비디오 파일을 열 수 없습니다: {video_path}")
        return None, None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[오류] 첫 프레임을 읽지 못했습니다.")
        return None, None

    temp_frame = frame.copy()
    cv2.namedWindow(drawing_window_name)
    cv2.setMouseCallback(drawing_window_name, draw_polygon_with_mouse)

    while True:
        display_frame = temp_frame.copy()

        # 첫 번째 사각형 (0~3)
        for i in range(min(4, len(draw_points))):
            cv2.circle(display_frame, draw_points[i], 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display_frame, draw_points[i-1], draw_points[i], (0, 255, 0), 2)
        if len(draw_points) >= 4:
            cv2.line(display_frame, draw_points[3], draw_points[0], (0, 255, 0), 2)

        # 두 번째 사각형 (4~7)
        if len(draw_points) > 4:
            for i in range(4, min(len(draw_points), 8)):
                cv2.circle(display_frame, draw_points[i], 5, (255, 0, 255), -1)
                if i > 4:
                    cv2.line(display_frame, draw_points[i-1], draw_points[i], (255, 0, 255), 2)
            if len(draw_points) == 8:
                cv2.line(display_frame, draw_points[7], draw_points[4], (255, 0, 255), 2)

        cv2.putText(display_frame, "1st Area: Green points", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(display_frame, "2nd Area: Magenta points", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
        cv2.imshow(drawing_window_name, display_frame)

        key = cv2.waitKey(20) & 0xFF
        if drawing_done and key != 255:
            break
        if key == ord('q'):
            break

    cv2.destroyWindow(drawing_window_name)

    if len(draw_points) < 8:
        print("영역 설정이 완료되지 않았습니다.")
        return None, None

    area_polygon1 = np.array(draw_points[0:4])
    area_polygon2 = np.array(draw_points[4:8])

    return area_polygon1, area_polygon2

def draw_transparent_polygon(frame, polygon, color=(0, 255, 255), alpha=0.2):
    overlay = frame.copy()
    pts = polygon.reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def polygon_center(polygon):
    cx = int(np.mean(polygon[:, 0]))
    cy = int(np.mean(polygon[:, 1]))
    return (cx, cy)

def show_video_with_boxes(video_path, frame_data, area_polygon1, area_polygon2, save_output=True, output_path="output.avi"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] 비디오 파일을 열 수 없습니다: {video_path}")
        return

    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 1

    prev_positions = {}
    area_times1 = {}
    area_times2 = {}
    
    event_msgs_area1 = ""
    event_msgs_area2 = ""

    def draw_polygon_lines(frame, polygon, color):
        pts = polygon.reshape((-1,1,2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 영역 선 및 반투명 채우기
        draw_polygon_lines(frame, area_polygon1, (0, 255, 255))
        draw_polygon_lines(frame, area_polygon2, (255, 0, 255))
        draw_transparent_polygon(frame, area_polygon1, color=(0, 255, 255), alpha=0.2)
        draw_transparent_polygon(frame, area_polygon2, color=(255, 0, 255), alpha=0.2)

        # 영역 번호 표시
        center1 = polygon_center(area_polygon1)
        center2 = polygon_center(area_polygon2)
        cv2.putText(frame, "1", center1, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, "2", center2, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
        
        if frame_idx in frame_data:
            for obj_id, x1, y1, x2, y2, label in frame_data[frame_idx]:
                color = LABEL_COLORS.get(label, DEFAULT_COLOR)
                label_name = LABEL_NAMES.get(label, f"Label:{label}")
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # 바운딩 박스 및 텍스트
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{obj_id}, {label_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                lat, lon = pixel_to_gps(cx, cy)
                cv2.circle(frame, (cx, cy), 3, color, -1)
                cv2.putText(frame, f"({lat:.6f}, {lon:.6f})", (cx + 5, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                inside1 = point_in_polygon((cx, cy), area_polygon1)
                inside2 = point_in_polygon((cx, cy), area_polygon2)

                # 영역1 체류 시간 계산
                if inside1:
                    if obj_id not in area_times1:
                        area_times1[obj_id] = {'start': frame_idx, 'end': frame_idx}
                    else:
                        area_times1[obj_id]['end'] = frame_idx
                else:
                    if obj_id in area_times1:
                        start = area_times1[obj_id]['start']
                        end = area_times1[obj_id]['end']
                        delta_frames = end - start
                        delta_time = delta_frames / fps
                        event_msgs_area1 = f"[LINE1 Crossed] ID:{obj_id} {delta_time:.2f}s"
                        del area_times1[obj_id]

                # 영역2 체류 시간 계산
                if inside2:
                    if obj_id not in area_times2:
                        area_times2[obj_id] = {'start': frame_idx, 'end': frame_idx}
                    else:
                        area_times2[obj_id]['end'] = frame_idx
                else:
                    if obj_id in area_times2:
                        start = area_times2[obj_id]['start']
                        end = area_times2[obj_id]['end']
                        delta_frames = end - start
                        delta_time = delta_frames / fps
                        event_msgs_area2 = f"[LINE2 Crossed] ID:{obj_id} {delta_time:.2f}s"

                        del area_times2[obj_id]

                prev_positions[obj_id] = (cx, cy)

            # 이벤트 메시지 화면 출력
            
            # 영역1 텍스트
            cv2.putText(frame, event_msgs_area1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # 테두리
            cv2.putText(frame, event_msgs_area1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

            # 영역2 텍스트
            cv2.putText(frame, event_msgs_area2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # 테두리
            cv2.putText(frame, event_msgs_area2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 200), 1)

        cv2.imshow("Video", frame)
        if save_output:
            out.write(frame)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    label_path = './assets/2024-10-21 08_56_19.337.txt'
    video_path = './assets/2024-10-21 08_56_19.337.mp4'


    frame_data = read_raw_data(label_path)

    # 1) 첫 프레임 띄우고 마우스로 2개 영역 그리기
    area_polygon1, area_polygon2 = get_area_polygons_from_user(video_path)
    if area_polygon1 is None or area_polygon2 is None:
        print("영역 설정 실패, 프로그램 종료")
        exit(1)

    # 2) 영역 그리기 완료 후, 영상 재생 및 영역 감지 시작
    show_video_with_boxes(video_path, frame_data, area_polygon1, area_polygon2, save_output=False)
