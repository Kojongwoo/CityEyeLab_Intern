# 📁 pyQT.py
# PyQt5 기반 GUI 영상 분석 도구
# - 선 통과 카운트 기능 (교차선 감지)
# - 정지 감지 기반 불법주정차 판별 기능 (ROI 체류 시간)
# - CSV 로그 저장 및 영상 상 시각화

# 작성자: (허종우)
# 최종 수정일: 2025-07-07

import sys, cv2, os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel,
    QHBoxLayout, QVBoxLayout, QPushButton
)
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QKeyEvent, QPainter, QPen, QFont
from PyQt5.QtWidgets import QInputDialog, QFileDialog, QMessageBox
from datetime import datetime
from utils import point_in_polygon

# 로그 폴더 없으면 생성
if not os.path.exists("logs"):
    os.makedirs("logs")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

LABEL_COLORS = {
    0: (0, 255, 0),      # Green
    1: (0, 0, 255),      # Red
    2: (255, 0, 0),      # Blue
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

def pixel_to_gps(x, y):
    gps1 = (37.401383, 127.112679)
    gps2 = (37.401371, 127.113207)
    px1 = (97, 415)
    px2 = (1342, 452)

    ratio = (x - px1[0]) / (px2[0] - px1[0])
    lat = gps1[0] + ratio * (gps2[0] - gps1[0])
    lon = gps1[1] + ratio * (gps2[1] - gps1[1])
    return (lat, lon)

# 두 선분이 교차하는지 판단하는 함수 (ccw 알고리즘 사용)
def crossed_line(p1, p2, prev_pt, curr_pt):
    # QPoint → 튜플로 변환
    A = (prev_pt.x(), prev_pt.y())
    B = (curr_pt.x(), curr_pt.y())
    C = (p1.x(), p1.y())
    D = (p2.x(), p2.y())

    def ccw(X, Y, Z):
        return (Z[1] - X[1]) * (Y[0] - X[0]) > (Y[1] - X[1]) * (Z[0] - X[0])
    
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

class VideoWindow(QWidget):

    # def __init__(self, video_path):
    def __init__(self, video_path, label_path):
        super().__init__()
        self.setWindowTitle("TrafficTool")
        
        # ✅ 1. 전체 PyQt 창 크기 고정
        window_width = 1920
        window_height = 1000

        self.line_number = 1
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

        # 영상 QLabel 크기 고정
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self.handle_mouse_press  # ✅ 마우스 클릭 이벤트 등록
        
        # 영상 QLabel 크기 = 전체 창의 90%
        video_width = int(window_width * 0.9)
        video_height = int(window_height * 0.9)
        self.video_label.setFixedSize(video_width, video_height)
        
        self.right_panel = QWidget(self)
        self.right_layout = QVBoxLayout()
        self.right_layout.setAlignment(Qt.AlignTop)  # 핵심: 위로 정렬
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(6)

        self.right_panel.setLayout(self.right_layout)

        self.label_path = label_path
        # self.label_path = './assets/2024-10-21 08_56_19.337.txt' # 라벨 경로 수정
        # self.label_path = './assets/2024-10-21 13_13_50.136.txt'  # 라벨 경로 수정
        self.frame_data = read_raw_data(self.label_path)
        self.frame_idx = 1  # 프레임 번호 추적

        # 닫기 버튼
        self.close_button = QPushButton("닫기", self)
        self.close_button.clicked.connect(self.close)

        # 수평 레이아웃: 왼쪽 영상 + 오른쪽 빈 영역
        hbox = QHBoxLayout()
        hbox.addWidget(self.video_label, 8)  
        hbox.addWidget(self.right_panel, 2)  

        # 수직 레이아웃: 영상 + 버튼
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.close_button)

        self.setLayout(vbox)

        # 영상 타이머 초기화 (정지 상태)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        # self.timer.start(30) # 초기에는 멈춘 상태!

        # 선 그리기 관련 변수
        self.drawing_enabled = True
        self.temp_points = []  # 두 점을 담을 임시 리스트
        self.lines = []        # [(p1, p2, line_number, description)] 형태로 선 저장
        
        self.undo_stack = []   
        self.redo_stack = []  # 되돌리기에 사용될 스택

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 불법주정차 결과 저장용 csv 초기화
        self.output_csv = f"./logs/illegal_parking_{timestamp}.csv"

        self.csv_header_written = False  # CSV 헤더를 1번만 쓰기 위한 플래그


        if os.path.exists(self.output_csv):
            os.remove(self.output_csv)

        # with open(self.output_csv, "w") as f:
        #     f.write("frame,obj_id,label,x1,y1,x2,y2,stop_seconds\n")

        # ⏯ 영상 첫 프레임 미리 표시
        self.show_first_frame()

        # self.showMaximized()
        self.installEventFilter(self)

        # 선 통과 여부 저장용 딕셔너리 추가
        self.cross_log = set()  # (obj_id, line_id) → 통과 여부

        self.line_labels = {} # line_id → QLabel 매핑



        # Undo, Redo 버튼 추가
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.undo_button.clicked.connect(self.handle_undo)
        self.redo_button.clicked.connect(self.handle_redo)
        self.right_layout.addWidget(self.undo_button)
        self.right_layout.addWidget(self.redo_button)

        # 일시정지 상태 추적
        self.is_paused = False  

        # 차량 정차 시간, 선 통과 이력 등 추적용 변수 초기화
        self.prev_positions = {}    # 각 객체의 이전 프레임 위치
        self.line_counts = {}       # 선별 카운트 저장 (몇 대가 통과했는지)
        self.crossed_lines = set()  # 중복 통과 방지용 (obj_id, line_id)
        self.illegal_log = set()    # 이미 불법정차로 기록된 차량 ID
        self.stop_watch = {}        # 객체별 ROI 체류 시간 추적


        # 선 모드 / 영역 모드 전환
        self.draw_mode = 'line'  # 또는 'area'
        self.temp_points = []    # 클릭한 점들을 여기에 저장

        self.stop_polygons = []  # → [ ([QPoint, QPoint, QPoint, QPoint], "설명"), ... ]

        self.line_mode_button = QPushButton("선 모드")
        self.area_mode_button = QPushButton("영역 모드")

        self.line_mode_button.setCheckable(True)
        self.area_mode_button.setCheckable(True)

        self.line_mode_button.setChecked(True)  # 기본은 선 모드

        self.line_mode_button.clicked.connect(self.set_line_mode)
        self.area_mode_button.clicked.connect(self.set_area_mode)

        self.right_layout.addWidget(self.line_mode_button)
        self.right_layout.addWidget(self.area_mode_button)

        shortcut_label = QLabel("🔑 단축키 안내:\n"
                        "Enter: 영상 재생/일시정지\n"
                        "R: 초기화 (리셋)\n"
                        "Q: 종료")
        shortcut_label.setStyleSheet("color: gray; font-size: 14px;")
        self.right_layout.addWidget(shortcut_label)

    def get_line_description(self, line_id):
        for p1, p2, num, desc in self.lines:
            if num == line_id:
                return desc
        return ""

    def inside_for_last_n_frames(self, obj_id, n=10):
    # """객체가 최근 n프레임 이상 ROI 내에 있었는지"""
        if obj_id in self.stop_watch:
            start = self.stop_watch[obj_id]['start']
            end = self.stop_watch[obj_id]['end']
            return (end - start) >= n
        return False

    def recently_crossed_line(self, obj_id):
        return any(obj == obj_id for obj, _ in self.cross_log)

    def is_within_violation_time(self, now):
        # """단속 시간대 여부 (08:00~20:00)"""
        return 8 <= now.hour < 20

    def is_illegal_vehicle_type(self, label):
        # """불법주정차 대상 차량인지"""
        exempt_types = ['police', 'ambulance']  # 예외 차량
        label_name = LABEL_NAMES.get(label, '')
        return label_name in ['car', 'bus_s', 'bus_m', 'truck_s', 'truck_m', 'truck_x', 'bike']
    
    def show_first_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 첫 프레임으로 되돌리기
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.update_display_with_lines()

    def update_frame(self):
        # 일시정지 상태 또는 선/영역 그리기 중일 경우 프레임 처리 중단
        if self.is_paused or self.drawing_enabled:
            return  
        # 다음 프레임 읽기
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        # BGR → RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.frame_idx in self.frame_data:
            # 현재 프레임의 객체 정보 처리
            for obj_id, x1, y1, x2, y2, label in self.frame_data[self.frame_idx]:
                color = LABEL_COLORS.get(label, DEFAULT_COLOR)
                label_name = LABEL_NAMES.get(label, f"Label:{label}")

                # 바운딩 박스 및 테스트
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                # 객체 ID + 라벨명
                cv2.putText(frame_rgb, f"ID:{obj_id}, {label_name}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 중심 좌표 계산 및 시각화
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(frame_rgb, (cx, cy), 3, color, -1)

                # GPS 좌표 표시
                lat, lon = pixel_to_gps(cx, cy)
                cv2.putText(frame_rgb, f"({lat:.6f}, {lon:.6f})", (cx + 5, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                curr_point = QPoint(cx, cy)


                # 선 통과 감지: 이전 위치와 현재 위치가 선을 가로질렀는지 확인 / 통과한 선은 crossed_lines에 추가
                if obj_id in self.prev_positions:
                    prev_point = self.prev_positions[obj_id]
                    for p1, p2, num, _ in self.lines:
                        if (obj_id, num) not in self.crossed_lines:
                            if crossed_line(p1, p2, prev_point, curr_point):
                                self.crossed_lines.add((obj_id, num))
                                self.line_counts[num] = self.line_counts.get(num, 0) + 1
                                print(f"🚗 차량 {obj_id} 선 {num} 통과 (총 {self.line_counts[num]}회)")

                                # 선 통과 기록
                                self.cross_log.add((obj_id, num))

                # 현재 위치 저장
                self.prev_positions[obj_id] = curr_point

            if hasattr(self, 'stop_polygons'):
                # 정지 감지 및 불법주정차 판단: 현재 위치가 사각형 영역 내에 있는지 확인
                for polygon in self.stop_polygons:
                    if len(polygon) == 4 and point_in_polygon((cx, cy), polygon):
                        # 현재 객체가 정지 감지 영역에 있는 경우
                        self.stop_watch.setdefault(obj_id, {'start': self.frame_idx, 'end': self.frame_idx, 'prev_pos': curr_point})
                        self.stop_watch[obj_id]['end'] = self.frame_idx
                        self.stop_watch[obj_id]['prev_pos'] = curr_point
                        break
                else:
                    if obj_id in self.stop_watch:
                        # ROI 벗어난 경우 총 체류시간 계산
                        start = self.stop_watch[obj_id]['start']
                        end = self.stop_watch[obj_id]['end']
                        seconds = (end - start) / self.fps
                       
                        prev_point = self.stop_watch[obj_id].get('prev_pos', curr_point)
                        move_dist = (curr_point - prev_point).manhattanLength()

                        # 불법 정차 감지: 5초 이상 정지 + 이동 거리 10픽셀 이하
                        # 불법 주정차로 감지된 차량은 콘솔 출력, 영상 위 경고 텍스트 표시, csv 파일에 로그 기록
                        now = datetime.now()
                        if (
                            seconds >= 8 and
                            move_dist < 10 and
                            self.inside_for_last_n_frames(obj_id, 10) and
                            not self.recently_crossed_line(obj_id) and
                            self.is_within_violation_time(now) and
                            self.is_illegal_vehicle_type(label)
                        ):
                            if obj_id not in self.illegal_log:
                                print(f"🚨 차량 {obj_id} ROI 내 불법정차 {seconds:.1f}초")
                                cv2.putText(frame_rgb, f"🚨 정차 차량 {obj_id}", (x1, y1 - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                self.illegal_log.add(obj_id)
                                with open(self.output_csv, "a", newline='') as f:
                                    f.write(f"{self.frame_idx},{obj_id},{label_name},{x1},{y1},{x2},{y2},{round(seconds,1)}\n")

                        del self.stop_watch[obj_id]

            # 프레임 저장 및 표시 갱신
            self.frame = frame_rgb

            # 현재 프레임 객체들의 선 통과 여부 기록
            for obj in self.frame_data.get(self.frame_idx, []):
                obj_id, x1, y1, x2, y2, label = obj
                base_info = [self.frame_idx, obj_id, x1, y1, x2, y2, label] # LABEL_NAMES.get(label, label): 라벨명 그대로 출력

                # 선 통과 여부
                line_states = []
                for i in range(1, self.line_number):  # 선 번호는 1부터 시작
                    state = 1 if (obj_id, i) in self.cross_log else 0
                    line_states.append(state)

                # 중심 좌표
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # 영역 포함 여부
                area_states = []
                for polygon in self.stop_polygons:
                    if len(polygon) == 4:
                        inside = point_in_polygon((cx, cy), polygon)
                        area_states.append(1 if inside else 0)
                    else:
                        area_states.append(0)

                # ⏬ CSV 헤더는 1번만 작성
                if not self.csv_header_written:
                    with open(self.output_csv, "w") as f:
                        base = "frame,obj_id,x1,y1,x2,y2,label"
                        for i in range(1, self.line_number):
                            base += f",line_{i}"
                        for j in range(1, len(self.stop_polygons) + 1):
                            base += f",area_{j}"
                        f.write(base + "\n")
                    self.csv_header_written = True

                with open(self.output_csv, "a", newline='') as f:
                    row = base_info + line_states + area_states
                    f.write(','.join(map(str, row)) + "\n")

            # # ✅ 선 통과 카운트 라벨 갱신
            for line_id, label in self.line_labels.items():
                count = self.line_counts.get(line_id, 0)
                label.setText(f"선 {line_id} ({self.get_line_description(line_id)}): Count: {count}")

            self.update_display_with_lines()
            self.frame_idx += 1

    def set_line_mode(self):
        self.draw_mode = 'line'
        self.line_mode_button.setChecked(True)
        self.area_mode_button.setChecked(False)
        self.temp_points.clear()

    def set_area_mode(self):
        self.draw_mode = 'area'
        self.line_mode_button.setChecked(False)
        self.area_mode_button.setChecked(True)
        self.temp_points.clear()
                           
    def update_display_with_lines(self):
        h, w, ch = self.frame.shape
        bytes_per_line = ch * w
        qimg = QImage(self.frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # 선 그리기
        painter = QPainter(pixmap)

        # 펜
        pen = QPen(Qt.red, 3)
        painter.setPen(pen)

        # 글꼴 크기 조정
        font = QFont()
        font.setPointSize(15)
        painter.setFont(font)

        # 선, 번호 그리기
        for p1, p2, num, desc in self.lines:
            painter.drawLine(p1, p2)
            mid_x = int((p1.x() + p2.x()) / 2)
            mid_y = int((p1.y() + p2.y()) / 2)
            painter.drawText(mid_x + 5, mid_y - 5, str(num))

            # count = self.line_counts.get(num, 0)
            # painter.drawText(mid_x + 5, mid_y + 50, f"Count: {count}")

            if desc:
                painter.drawText(mid_x + 5, mid_y + 30, desc)  # 설명 (조금 아래로)

        # 영역(사각형) 그리기 + 설명 표시
        if hasattr(self, 'stop_polygons'):
            for i, (polygon, desc) in enumerate(self.stop_polygons):
                if len(polygon) == 4:
                    pen = QPen(Qt.yellow, 2, Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawPolygon(*polygon)
                    for pt in polygon:
                        painter.drawEllipse(pt, 4, 4)

                    # 중심 좌표 계산
                    cx = sum([pt.x() for pt in polygon]) // 4
                    cy = sum([pt.y() for pt in polygon]) // 4

                    # 설명 텍스트 출력
                    painter.drawText(cx + 5, cy - 5, f"{i+1}. {desc}")


        for pt in self.temp_points:
            painter.setPen(QPen(Qt.green, 2))
            painter.drawEllipse(pt, 5, 5)

        painter.end()

        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation
        ))

    def handle_mouse_press(self, event):
        if self.drawing_enabled and event.button() == Qt.LeftButton:
            # 클릭 위치 (video_label 기준 좌표)
            label_pos = event.pos()

            # QLabel과 실제 프레임 크기 비교해서 비율 계산
            label_width = self.video_label.width()
            label_height = self.video_label.height()
            frame_height, frame_width, _ = self.frame.shape

            scale_x = frame_width / label_width
            scale_y = frame_height / label_height

            # 좌표 보정
            corrected_x = int(label_pos.x() * scale_x)
            corrected_y = int(label_pos.y() * scale_y)
            corrected_point = QPoint(corrected_x, corrected_y)

            self.temp_points.append(corrected_point)

            # 선 모드일 경우: 점 2개 찍으면 하나의 선 생성
            if self.draw_mode == 'line' and len(self.temp_points) == 2:
                existing_ids = [line[2] for line in self.lines]
                new_id = max(existing_ids, default=0) + 1

                text, ok = QInputDialog.getText(self, f"선 {new_id} 설명 입력", "이 선에 대한 설명을 입력하세요:")

            # 👉 설명 입력 받기
                if ok and text.strip(): 
                    description = text.strip()

                    line_obj = (self.temp_points[0], self.temp_points[1], new_id, description)
                    self.lines.append(line_obj)

                    # 👉 오른쪽 Count 표시 라벨 생성
                    label = QLabel(f"선 {new_id} ({description}): Count: 0")
                    label.setStyleSheet("color: darkred; font-size: 14px;")
                    self.right_layout.addWidget(label)
                    self.line_labels[new_id] = label
                    label.setWordWrap(True)

                    self.undo_stack.append(("line", line_obj))  # 🔥 추가!
                    self.line_number += 1
                    self.redo_stack.clear()
                else:
                    print("설명이 입력되지 않아 선이 생성되지 않습니다.")
                    self.temp_points.clear()
                    self.update_display_with_lines()             # 💡 화면 갱신
                    return
                self.temp_points.clear()
                self.update_display_with_lines()

            # 영역 모드일 경우: 점 4개 찍으면 사각형 ROI 생성
            elif self.draw_mode == 'area' and len(self.temp_points) == 4:
                polygon = self.temp_points.copy()

                # ✨ 설명 입력 받기
                text, ok = QInputDialog.getText(self, f"영역 {len(self.stop_polygons)+1} 설명", "이 영역에 대한 설명을 입력하세요:")
                
                if ok and text.strip():
                    description = text.strip()
                    area_obj = (polygon, description)
                    self.stop_polygons.append(area_obj)
                    self.undo_stack.append(("area", area_obj))  # 🔥 추가!
                    print(f"🚧 정지 감지 영역 {len(self.stop_polygons)} 생성 완료")
                else:
                    print("설명이 입력되지 않아 영역이 생성되지 않습니다.")
                    self.temp_points.clear()
                    self.update_display_with_lines()
                
                self.temp_points.clear()
            self.update_display_with_lines()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Q:
            print("Q 키 눌림: 종료")
            self.close()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.drawing_enabled:
                print("Enter 키 눌림: 영상 재생 시작")
                self.drawing_enabled = False
                self.timer.start(30)
            else: # ⏯ 재생 중일 때 Enter로 일시정지 / 재개
                self.is_paused = not self.is_paused
                print("⏸ 일시정지" if self.is_paused else "▶ 재생")

        elif event.key() == Qt.Key_R:
            print("🔁 R 키: 상태 초기화")

            self.lines.clear()
            self.temp_points.clear()
            self.stop_polygons = [] # 영역 여러개 저장
            self.line_counts.clear()
            self.crossed_lines.clear()
            self.stop_watch.clear()
            self.prev_positions.clear()
            self.line_number = 1
            self.drawing_enabled = True
            self.draw_mode = 'line'
            self.line_mode_button.setChecked(True)
            self.area_mode_button.setChecked(False)
            self.cross_log.clear()  # ⏬ 선 통과 상태 초기화

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 1
            ret, frame = self.cap.read()
            if ret:
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.update_display_with_lines()
            self.timer.stop()

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

    def handle_undo(self):
        if not self.undo_stack:
            return
        last_type, last_obj = self.undo_stack.pop()
        self.redo_stack.append((last_type, last_obj))

        if last_type == "line":
            if last_obj in self.lines:
                self.lines.remove(last_obj)
                line_id = last_obj[2]

                if line_id in self.line_labels:
                    label = self.line_labels.pop(line_id)
                    self.right_layout.removeWidget(label)
                    label.deleteLater()

        elif last_type == "area":
            if last_obj in self.stop_polygons:
                self.stop_polygons.remove(last_obj)

        self.update_display_with_lines()

    def handle_redo(self):
        if not self.redo_stack:
            return
        item_type, item = self.redo_stack.pop()
        self.undo_stack.append((item_type, item))

        if item_type == "line":
            self.lines.append(item)
        elif item_type == "area":
            self.stop_polygons.append(item)

        self.update_display_with_lines()

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     video_path = './assets/2024-10-21 08_56_19.337.mp4'  # 영상 경로 수정
#     # video_path = './assets/2024-10-21 13_13_50.136.mp4'  # 영상 경로 수정
#     window = VideoWindow(video_path)
#     window.show()
#     sys.exit(app.exec_())

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # ✅ 영상 파일 선택
    video_path, _ = QFileDialog.getOpenFileName(
        None,
        "영상 파일 선택",
        "./assets",
        "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
    )
    if not video_path:
        QMessageBox.warning(None, "경고", "영상 파일을 선택하지 않으면 프로그램이 종료됩니다.")
        sys.exit()

    # ✅ 라벨 파일 선택
    label_path, _ = QFileDialog.getOpenFileName(
        None,
        "라벨 텍스트 파일 선택",
        "./assets",
        "Text Files (*.txt);;All Files (*)"
    )
    if not label_path:
        QMessageBox.warning(None, "경고", "라벨 파일을 선택하지 않으면 프로그램이 종료됩니다.")
        sys.exit()

    # ✅ 창 열기
    window = VideoWindow(video_path, label_path)
    window.show()
    sys.exit(app.exec_())