# 📁 pyQT.py
# PyQt5 기반 GUI 영상 분석 도구
# - 선 통과 카운트 기능 (교차선 감지)
# - 정지 감지 기반 불법주정차 판별 기능 (ROI 체류 시간)
# - CSV 로그 저장 및 영상 상 시각화

# 작성자: (허종우)
# 최종 수정일: 2025-07-08

import sys, cv2, os, copy
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel,
    QHBoxLayout, QVBoxLayout, QPushButton, QInputDialog,
    QFileDialog, QMessageBox,  QSlider, QLineEdit, QComboBox
)
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QKeyEvent, QPainter, QPen, QFont, QBrush, QColor
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

def get_location_folder_key(path):
    return os.path.basename(os.path.dirname(path))

def read_raw_data(path, frame_offset=0):
    frame_data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = list(map(int, line.split(',')))
            frame, obj_id, x1, y1, x2, y2, label = values
            # frame += frame_offset  # ✅ 누적 프레임 반영
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

# def draw_transparent_polygon(frame, polygon, color=(0, 255, 0), alpha=0.2):
#     overlay = frame.copy()
#     pts = np.array([[pt.x(), pt.y()] for pt in polygon], np.int32)
#     pts = pts.reshape((-1, 1, 2))
#     cv2.fillPoly(overlay, [pts], color)
#     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_qt_transparent_polygon(painter, polygon, color=Qt.green, alpha=80):
    color_with_alpha = QColor(color)
    color_with_alpha.setAlpha(alpha)
    brush = QBrush(color_with_alpha)
    painter.setBrush(brush)

    pts = [pt for pt in polygon]
    painter.drawPolygon(*pts)

    painter.setBrush(Qt.NoBrush)  # 그 후 다시 원래대로 되돌림

class VideoWindow(QWidget):

    # def __init__(self, video_path):
    def __init__(self, video_label_pairs):
        super().__init__()
        self.setWindowTitle("TrafficTool")
        self.video_label_pairs = video_label_pairs  # 전체 쌍
        self.current_index = 0

        video_path, label_path = self.video_label_pairs[self.current_index]
        self.video_path = video_path
        self.label_path = label_path
        
        # ✅ 1. 전체 PyQt 창 크기 고정
        window_width = 1800
        window_height = 900

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
        video_width = int(window_width * 0.8)
        video_height = int(window_height * 0.95)
        self.video_label.setFixedSize(video_width, video_height)
        
        self.right_panel = QWidget(self)
        self.right_panel.setFixedWidth(360)  # 너비 고정 (원하는 값)
        self.right_layout = QVBoxLayout()
        self.right_layout.setAlignment(Qt.AlignTop)  # 핵심: 위로 정렬
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(12)

        self.right_panel.setLayout(self.right_layout)

        # GUI 상단에 QComboBox 추가
        self.file_selector = QComboBox()
        for v, l in self.video_label_pairs:
            name = os.path.basename(v)
            self.file_selector.addItem(name)

        self.file_selector.currentIndexChanged.connect(self.change_file)
        self.right_layout.addWidget(self.file_selector)

        # ⬇ 현재 영상 제목을 표시할 QLabel 추가
        self.video_name_label = QLabel(f"🎬 현재 영상: {os.path.basename(self.video_path)}")
        self.video_name_label.setWordWrap(True)                      # ✅ 줄바꿈 허용
        self.video_name_label.setMaximumWidth(320)                   # ✅ 적당한 최대 너비 지정
        self.video_name_label.setStyleSheet("font-size: 18px; font-weight: bold; color: navy;")
        self.right_layout.addWidget(self.video_name_label)


        self.label_path = label_path
        # self.label_path = './assets/2024-10-21 08_56_19.337.txt' # 라벨 경로 수정
        # self.label_path = './assets/2024-10-21 13_13_50.136.txt'  # 라벨 경로 수정
        self.cumulative_frame_offset = 0  # 누적 프레임 오프셋
        self.frame_data = read_raw_data(self.label_path)
        self.frame_idx = 1  # 프레임 번호 추적

        self.per_file_states = {}  # 각 영상별 상태 저장용 딕셔너리

        # 닫기 버튼
        self.close_button = QPushButton("닫기", self)
        self.close_button.clicked.connect(self.close)

        # 수평 레이아웃: 왼쪽 영상 + 오른쪽 빈 영역
        hbox = QHBoxLayout()
        hbox.setSpacing(20)  # ← 영상과 오른쪽 패널 사이 간격 설정
        hbox.addWidget(self.video_label, stretch = 0)  
        hbox.addWidget(self.right_panel)  

        # 수직 레이아웃: 영상 + 버튼
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.setAlignment(hbox, Qt.AlignTop)  # ✅ 위쪽 정렬 추가
        vbox.addWidget(self.close_button)

        self.setLayout(vbox)

        # 영상 타이머 초기화 (정지 상태)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 선 그리기 관련 변수
        self.drawing_enabled = True
        self.temp_points = []  # 두 점을 담을 임시 리스트
        self.lines = []        # [(p1, p2, line_number, description)] 형태로 선 저장

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 불법주정차 결과 저장용 csv 초기화
        video_date_str = os.path.basename(self.video_path).split()[0]  # "2024-10-21"
        
        self.output_csv = f"./logs/{video_date_str}_analyzed.csv"
        if os.path.exists(self.output_csv):
            os.remove(self.output_csv)  # 기존 파일 삭제 (처음만 실행)
        self.csv_header_written = False

        # 예: 2024-07-11_analyzed.csv, v2, v3 ...
        # video_date_str = os.path.basename(self.video_path).split()[0]  # "2024-10-21"
        today_str = datetime.now().strftime("%Y-%m-%d")
        base_name = f"{video_date_str}_analyzed"
        csv_dir = "./logs"

        # 파일명 중복 방지: v2, v3, ...
        version = 1
        while True:
            if version == 1:
                csv_path = os.path.join(csv_dir, f"{base_name}.csv")
            else:
                csv_path = os.path.join(csv_dir, f"{base_name}_v{version}.csv")
            if not os.path.exists(csv_path):
                break
            version += 1

        self.output_csv = csv_path
        self.csv_header_written = False


        # ⏯ 영상 첫 프레임 미리 표시
        self.show_first_frame()

        self.installEventFilter(self)

        # 선 통과 여부 저장용 딕셔너리 추가
        self.cross_log = set()  # (obj_id, line_id) → 통과 여부

        self.line_labels = {} # line_id → QLabel 매핑

        # 선/버튼과 ID 간 매핑 구조 추가
        self.line_widgets = {}  # line_id → QWidget
        self.area_labels = {}
        self.area_widgets = {}
        self.area_number = 1

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

        self.line_mode_button = QPushButton("선 만들기")
        self.area_mode_button = QPushButton("영역 만들기")
        self.line_mode_button.setFixedHeight(40)   # 기본은 25~30, 이건 더 큼직
        self.area_mode_button.setFixedHeight(40)
        self.line_mode_button.setStyleSheet("font-size: 14px;")
        self.area_mode_button.setStyleSheet("font-size: 14px;")
        self.line_mode_button.setCheckable(True)
        self.area_mode_button.setCheckable(True)
 

        self.line_mode_button.setChecked(True)  # 기본은 선 모드

        self.line_mode_button.clicked.connect(self.set_line_mode)
        self.area_mode_button.clicked.connect(self.set_area_mode)

        self.right_layout.addWidget(self.line_mode_button)
        self.right_layout.addWidget(self.area_mode_button)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        

        # shortcut_label = QLabel("🔑 단축키 안내:\n"
        #                 "Enter: 영상 재생/일시정지\n"
        #                 "A: 이전 프레임\n"
        #                 "D: 다음 프레임\n"
        #                 "R: 초기화 (리셋)\n"
        #                 "Q: 종료")

        # shortcut_label.setMinimumHeight(120)  # 🔥 이게 핵심!
        # shortcut_label.setStyleSheet("color: black; font-size: 24px; line-height: 160%;")
        # self.right_layout.addWidget(shortcut_label)

        # ⏯ 영상 재생 / 일시정지 버튼
        self.play_pause_button = QPushButton("⏯ 재생 / 일시정지")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.right_layout.addWidget(self.play_pause_button)

        # ▶ 이전 프레임 버튼
        self.prev_frame_button = QPushButton("◀ 이전 프레임")
        self.prev_frame_button.clicked.connect(self.go_prev_frame)
        self.prev_frame_button.setFixedHeight(50)

        # ▶ 다음 프레임 버튼
        self.next_frame_button = QPushButton("▶ 다음 프레임")
        self.next_frame_button.clicked.connect(self.go_next_frame)
        self.next_frame_button.setFixedHeight(50)

        # ✅ 두 버튼을 한 줄로 묶기
        frame_nav_layout = QHBoxLayout()
        frame_nav_layout.addWidget(self.prev_frame_button)
        frame_nav_layout.addWidget(self.next_frame_button)
        self.right_layout.addLayout(frame_nav_layout)


        # 🔁 초기화 버튼
        self.reset_button = QPushButton("🔁 초기화")
        self.reset_button.clicked.connect(self.reset_video_state)
        self.right_layout.addWidget(self.reset_button)

        self.frame_label = QLabel("프레임: 1")
        self.frame_label.setStyleSheet("color: navy; font-size: 20px;")
        self.right_layout.addWidget(self.frame_label)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(self.total_frames)
        self.frame_slider.setValue(self.frame_idx)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.sliderReleased.connect(self.handle_slider_moved)
        self.right_layout.addWidget(self.frame_slider)

        self.play_pause_button.setFixedHeight(50)
        self.prev_frame_button.setFixedHeight(50)
        self.next_frame_button.setFixedHeight(50)
        self.reset_button.setFixedHeight(50)

        # 프레임 검색 입력창 + 버튼
        self.search_frame_input = QLineEdit()
        self.search_frame_input.setPlaceholderText("프레임 번호 입력")
        self.search_frame_input.setFixedWidth(120)

        self.search_frame_button = QPushButton("이동")
        self.search_frame_button.clicked.connect(self.jump_to_frame)

        # 수평으로 묶기
        frame_jump_layout = QHBoxLayout()
        frame_jump_layout.addWidget(self.search_frame_input)
        frame_jump_layout.addWidget(self.search_frame_button)
        self.right_layout.addLayout(frame_jump_layout)

        self.force_draw_objects = False  # 👉 객체 박스 강제 그리기 용도

        self.group_states = {}  # 👉 폴더별(장소별) 상태 저장

        # 전체 선/영역 개수를 저장하는 변수
        self.max_line_number = 0
        self.max_area_number = 0

        # ✅ 누적 최대값 (영상 전체 기준)
        self.global_max_line_number = 0
        self.global_max_area_number = 0


    def toggle_play_pause(self):
        if self.drawing_enabled:
            self.drawing_enabled = False
            self.is_paused = False
            self.timer.start(30)
        else:
            self.is_paused = not self.is_paused

    def go_prev_frame(self):
        if self.frame_idx <= 1:
            print("첫 프레임입니다.")
            return
        self.frame_idx -= 1
        frame = self.safe_seek(self.frame_idx)
        if frame is not None:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_label.setText(f"프레임: {self.frame_idx}")
            self.frame_slider.setValue(self.frame_idx)
            self.force_draw_objects = True
            self.update_frame()
            self.force_draw_objects = False

    def go_next_frame(self):
        if self.frame_idx >= self.total_frames:
            print("마지막 프레임입니다.")
            return
        self.frame_idx += 1
        frame = self.safe_seek(self.frame_idx)
        if frame is not None:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_label.setText(f"프레임: {self.frame_idx}")
            self.frame_slider.setValue(self.frame_idx)
            self.force_draw_objects = True
            self.update_frame()
            self.force_draw_objects = False

    def reset_video_state(self):
        print("🔁 상태 초기화")
        self.lines.clear()
        self.temp_points.clear()
        self.stop_polygons = []
        self.line_counts.clear()
        self.crossed_lines.clear()
        self.stop_watch.clear()
        self.prev_positions.clear()
        self.line_number = 1
        self.drawing_enabled = True
        self.draw_mode = 'line'
        self.line_mode_button.setChecked(True)
        self.area_mode_button.setChecked(False)
        self.cross_log.clear()
        self.area_number = 1

        # 위젯 정리
        for widget in self.line_widgets.values():
            self.right_layout.removeWidget(widget)
            widget.deleteLater()
        self.line_widgets.clear()

        for widget in self.area_widgets.values():
            self.right_layout.removeWidget(widget)
            widget.deleteLater()
        self.area_widgets.clear()

        self.line_labels.clear()
        self.area_labels.clear()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 1
        self.frame_label.setText("프레임: 1")
        self.frame_slider.setValue(1)
        self.search_frame_input.clear()

        ret, frame = self.cap.read()
        if ret:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.update_display_with_lines()
        self.timer.stop()
        
    def change_file(self, index):
        print(f"📦 현재 영상: {self.video_path}")
        print(f"📄 매칭된 라벨: {self.label_path}")
        print(f"📊 라벨 데이터 프레임 수: {len(self.frame_data)}")  # 이게 0이면 라벨 없음

        # 👉 현재 상태 저장
        self.per_file_states[self.video_path] = {
            "frame_idx": self.frame_idx,
            "lines": copy.deepcopy(self.lines),
            "stop_polygons": copy.deepcopy(self.stop_polygons),
            "line_counts": copy.deepcopy(self.line_counts),
            "crossed_lines": copy.deepcopy(self.crossed_lines),
            "stop_watch": copy.deepcopy(self.stop_watch),
            "illegal_log": copy.deepcopy(self.illegal_log),
            "prev_positions": copy.deepcopy(self.prev_positions),
            "line_number": self.line_number,
            "area_number": self.area_number,
        }

        # 👉 현재 영상/장소 그룹 상태 저장
        group_key = get_location_folder_key(self.video_path)
        self.group_states[group_key] = {
            "lines": copy.deepcopy(self.lines),
            "stop_polygons": copy.deepcopy(self.stop_polygons),
            "line_number": self.line_number,
            "area_number": self.area_number,
}
        self.current_index = index
        # 누적 프레임 오프셋 계산 (index 이전 영상들의 총 프레임 수)
        self.cumulative_frame_offset = 0
        for i in range(index):
            prev_video_path, _ = self.video_label_pairs[i]
            cap = cv2.VideoCapture(prev_video_path)
            total_prev = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.cumulative_frame_offset += total_prev

        # 라벨 로딩 시 누적 프레임 오프셋 반영
        self.video_path, self.label_path = self.video_label_pairs[index]
        self.frame_data = read_raw_data(self.label_path, frame_offset=self.cumulative_frame_offset)

        # ✅ 여기 아래에 영상 이름 라벨 갱신 추가!
        self.video_name_label.setText(f"🎬 현재 영상: {os.path.basename(self.video_path)}")

        # 영상 재로딩
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        # self.frame_data = read_raw_data(self.label_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_slider.setMaximum(self.total_frames)
        # 👉 새 영상 상태 복원 or 초기화
        state = self.per_file_states.get(self.video_path, None)

        # # # ✅ 여기 바로 아래에 추가!
        # base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        # now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # self.output_csv = f"./logs/{base_name}_{now_str}_illegal_parking.csv"

        # self.csv_header_written = os.path.exists(self.output_csv)
        # ✅ 두 번째 영상부터도 헤더 다시 작성할 수 있도록 초기화
        # if os.path.exists(self.output_csv):
        #     os.remove(self.output_csv)
        # self.csv_header_written = False

        if state:
            self.frame_idx = state["frame_idx"]
            self.lines = state["lines"]
            self.stop_polygons = state["stop_polygons"]
            self.line_counts = state["line_counts"]

            self.crossed_lines = state["crossed_lines"]
            self.stop_watch = state["stop_watch"]
            self.illegal_log = state["illegal_log"]
            self.prev_positions = state["prev_positions"]
            self.line_number = state["line_number"]
            self.area_number = state["area_number"]
        else:
            self.frame_idx = 1
            self.lines = []
            self.stop_polygons = []
            self.line_counts = {}
            self.crossed_lines = set()
            self.stop_watch = {}
            self.illegal_log = set()
            self.prev_positions = {}
            self.line_number = 1
            self.area_number = 1

            # ✅ 추가: 같은 폴더 그룹 상태 복원
            group_key = get_location_folder_key(self.video_path)
            group_state = self.group_states.get(group_key, None)
            if group_state:
                self.lines = copy.deepcopy(group_state["lines"])
                self.stop_polygons = copy.deepcopy(group_state["stop_polygons"])
                self.line_number = group_state["line_number"]
                self.area_number = group_state["area_number"]

        # 👉 우측 라벨 및 위젯 초기화
        for widget in self.line_widgets.values():
            self.right_layout.removeWidget(widget)
            widget.deleteLater()
        self.line_widgets.clear()
        self.line_labels.clear()

        for widget in self.area_widgets.values():
            self.right_layout.removeWidget(widget)
            widget.deleteLater()
        self.area_widgets.clear()
        self.area_labels.clear()

        # 👉 선 다시 추가
        for p1, p2, line_id, desc in self.lines:
            label = QLabel(f"선 {line_id} ({desc}): Count: {self.line_counts.get(line_id, 0)}")
            label.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
            label.setWordWrap(True)        # ✅ 줄바꿈 허용
            edit_btn = QPushButton("수정")
            delete_btn = QPushButton("삭제")
            edit_btn.setFixedSize(60, 30)
            delete_btn.setFixedSize(60, 30)
            edit_btn.clicked.connect(lambda _, lid=line_id: self.edit_line_description(lid))
            delete_btn.clicked.connect(lambda _, lid=line_id: self.delete_line(lid))

            widget = QWidget()
            layout = QHBoxLayout()
            layout.addWidget(label)
            layout.addWidget(edit_btn)
            layout.addWidget(delete_btn)
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)

            self.right_layout.addWidget(widget)
            self.line_labels[line_id] = label
            self.line_widgets[line_id] = widget

        # 👉 영역 다시 추가
        for idx, (polygon, desc) in enumerate(self.stop_polygons):
            area_id = idx + 1
            label = QLabel(f"영역 {area_id} ({desc})")
            label.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
            label.setWordWrap(True)        # ✅ 줄바꿈 허용
            edit_btn = QPushButton("수정")
            delete_btn = QPushButton("삭제")
            edit_btn.setFixedSize(60, 30)
            delete_btn.setFixedSize(60, 30)
            edit_btn.clicked.connect(lambda _, aid=area_id: self.edit_area_description(aid))
            delete_btn.clicked.connect(lambda _, aid=area_id: self.delete_area(aid))

            widget = QWidget()
            layout = QHBoxLayout()
            layout.addWidget(label)
            layout.addWidget(edit_btn)
            layout.addWidget(delete_btn)
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)

            self.right_layout.addWidget(widget)
            self.area_labels[area_id] = label
            self.area_widgets[area_id] = widget

        # self.frame_idx = 1
        self.drawing_enabled = True
        self.cross_log.clear()
        self.prev_positions.clear()
        self.line_counts.clear()
        self.crossed_lines.clear()
        self.stop_watch.clear()
        self.illegal_log.clear()

        # self.lines.clear()
        # self.stop_polygons.clear()
        # self.line_number = 1
        # self.area_number = 1
        # ✅ 현재 선/영역 리스트에서 최대 ID를 기준으로 line_number, area_number 재설정
        existing_line_ids = [line[2] for line in self.lines]  # line = (p1, p2, line_id, desc)
        self.line_number = max(existing_line_ids, default=0) + 1
        self.area_number = len(self.stop_polygons) + 1

        self.update_display_with_lines()
        # self.show_first_frame()
        self.frame_slider.setValue(self.frame_idx)
        self.frame_label.setText(f"프레임: {self.frame_idx}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx - 1)
        ret, frame = self.cap.read()

        if ret:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.update_display_with_lines()
            self.frame_label.setText(f"프레임: {self.frame_idx}")
            self.frame_slider.setValue(self.frame_idx)

        # ✅ 누적된 전체 선/영역 개수를 유지하도록 max 업데이트
        self.max_line_number = max(self.max_line_number, len(self.lines))
        self.max_area_number = max(self.max_area_number, len(self.stop_polygons))

        # ✅ 누적 max 갱신
        self.global_max_line_number = max(self.global_max_line_number, self.max_line_number)
        self.global_max_area_number = max(self.global_max_area_number, self.max_area_number)

        # ✅ 여기에 이 2줄 추가
        self.drawing_enabled = True
        self.is_paused = True

    def get_line_description(self, line_id):
        for p1, p2, num, desc in self.lines:
            if num == line_id:
                return desc
        return ""
    
    def safe_seek(self, target_frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(target_frame - 1):
            ret, _ = self.cap.read()
            if not ret:
                print(f"[ERROR] Frame {i+1} read failed during seek")

        ret, frame = self.cap.read()
        if not ret:
            print(f"[ERROR] Frame {target_frame} read failed at target")
            return None

        if frame is None:
            print(f"[ERROR] Frame {target_frame} is None")
            return None

        return frame.copy()  # ✅ 반드시 frame 반환해야 정상 동작
    
    def jump_to_frame(self):
        text = self.search_frame_input.text().strip()
        if not text.isdigit():
            QMessageBox.warning(self, "입력 오류", "숫자만 입력해주세요.")
            return

        frame_number = int(text)
        if not (1 <= frame_number <= self.total_frames):
            QMessageBox.warning(self, "범위 오류", f"1 ~ {self.total_frames} 사이의 값을 입력해주세요.")
            return

        frame = self.safe_seek(frame_number)
        if frame is not None:
            self.frame_idx = frame_number
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_label.setText(f"프레임: {self.frame_idx}")
            self.frame_slider.setValue(self.frame_idx)
            self.update_display_with_lines()

            self.force_draw_objects = True
            self.update_frame()
            self.force_draw_objects = False

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
        # 영상 끝에 도달한 경우 먼저 확인
        if self.frame_idx > self.total_frames:
            self.timer.stop()
            print("🔚 현재 영상 종료됨")
            # 👉 다음 영상으로 넘어갈 수 있다면
            if self.current_index + 1 < len(self.video_label_pairs):
                self.current_index += 1
                self.change_file(self.current_index)
                self.drawing_enabled = False
                self.is_paused = False
                self.timer.start(30)
            else:
                print("✅ 모든 영상 재생 완료")
            return

        # 일시정지 상태 또는 선/영역 그리기 중일 경우 프레임 처리 중단
        if (self.is_paused or self.drawing_enabled) and not self.force_draw_objects:
            return 
        
        # 다음 프레임 읽기
        ret, frame = self.cap.read()

        if not ret:
            print("⚠️ 프레임 읽기 실패 → 다음 영상으로 전환 시도")

            if self.current_index + 1 < len(self.video_label_pairs):
                self.current_index += 1
                self.change_file(self.current_index)
                self.drawing_enabled = False
                self.is_paused = False
                self.timer.start(30)
            else:
                print("✅ 모든 영상 재생 완료")
                self.timer.stop()

            return
        # BGR → RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.frame = frame_rgb  # 💥 반드시 먼저 설정
       
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

            # if hasattr(self, 'stop_polygons'):
            #     # 각 ROI를 반투명으로 채움
            #     for polygon in self.stop_polygons:
            #         if len(polygon) == 4:
            #             draw_transparent_polygon(frame_rgb, polygon, color=(0, 128, 0), alpha=0.25)                

                # 정지 감지 및 불법주정차 판단:
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
                # real_frame = self.cumulative_frame_offset + self.frame_idx
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
                # if not self.csv_header_written:
                #     with open(self.output_csv, "w") as f:
                #         base = "video,frame,obj_id,x1,y1,x2,y2,label"
                #         for i in range(1, self.line_number):
                #             base += f",line_{i}"
                #         for j in range(1, len(self.stop_polygons) + 1):
                #             base += f",area_{j}"
                #         f.write(base + "\n")
                #     self.csv_header_written = True

                # ✅ 헤더 작성 전 항상 max 값 갱신
                self.max_line_number = max(self.max_line_number, len(self.lines))
                self.max_area_number = max(self.max_area_number, len(self.stop_polygons))

                # 🔁 누적 최대값도 갱신 (매 프레임 체크)
                self.global_max_line_number = max(self.global_max_line_number, self.max_line_number)
                self.global_max_area_number = max(self.global_max_area_number, self.max_area_number)

                if not self.csv_header_written:
                    with open(self.output_csv, "w") as f:
                        base = "video,frame,obj_id,x1,y1,x2,y2,label"
                        for i in range(1, self.global_max_line_number + 1):
                            base += f",line_{i}"
                        for j in range(1, self.global_max_area_number + 1):
                            base += f",area_{j}"
                        f.write(base + "\n")
                    self.csv_header_written = True

                with open(self.output_csv, "a", newline='') as f:
                    video_name = os.path.basename(self.video_path)
                    row = [video_name] + base_info + line_states + area_states
                    f.write(','.join(map(str, row)) + "\n")

            # # ✅ 선 통과 카운트 라벨 갱신
            for line_id, label in self.line_labels.items():
                count = self.line_counts.get(line_id, 0)
                label.setText(f"선 {line_id} ({self.get_line_description(line_id)}): Count: {count}")

        if not self.is_paused and not self.force_draw_objects:
            self.frame_idx += 1

        if self.frame_idx > self.total_frames:
            # 👉 다음 영상이 있다면 자동으로 전환
            if self.current_index + 1 < len(self.video_label_pairs):
                print(f"📂 영상 {self.current_index + 1} 재생 완료. 다음 영상으로 전환합니다.")
                self.current_index += 1
                self.change_file(self.current_index)

                self.drawing_enabled = False
                self.is_paused = False
                self.timer.start(30)  # 다음 영상 재생 계속
            else:
                print("✅ 모든 영상 재생 완료")
                self.timer.stop()
            return

        self.update_display_with_lines()
        self.frame_label.setText(f"프레임: {self.frame_idx}") # ✅ 현재 프레임 표시
        self.frame_slider.setValue(self.frame_idx)

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
        # print(f"[DRAW] Displaying frame {self.frame_idx}")
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
        font.setBold(True)
        painter.setFont(font)

        painter.drawText(20, 40, f"Frame: {self.frame_idx}")

        # 선, 번호 그리기
        for p1, p2, num, desc in self.lines:
            painter.drawLine(p1, p2)
            mid_x = int((p1.x() + p2.x()) / 2)
            mid_y = int((p1.y() + p2.y()) / 2)
            painter.drawText(mid_x + 5, mid_y - 5, str(num))

            if desc:
                painter.drawText(mid_x + 5, mid_y + 30, desc)  # 설명 (조금 아래로)

        # 영역(사각형) 그리기 + 반투명 채우기 + 설명 표시
        if hasattr(self, 'stop_polygons'):
            for i, (polygon, desc) in enumerate(self.stop_polygons):
                if len(polygon) == 4:
                    # ✅ 1) 반투명 채우기 먼저
                    draw_qt_transparent_polygon(painter, polygon, Qt.green, alpha=80)

                    # ✅ 2) 외곽선 그리기
                    pen = QPen(Qt.green, 2, Qt.SolidLine)
                    painter.setPen(pen)
                    painter.drawPolygon(*polygon)
                    for pt in polygon:
                        painter.drawEllipse(pt, 4, 4)

                    # ✅ 3) 텍스트 표시
                    cx = sum([pt.x() for pt in polygon]) // 4
                    cy = sum([pt.y() for pt in polygon]) // 4
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
                    label.setStyleSheet("color: red; font-size: 16px;; font-weight: bold;")
                    label.setWordWrap(True)  # 줄바꿈 설정
                    
                    edit_btn = QPushButton("수정")
                    delete_btn = QPushButton("삭제")
                    edit_btn.setFixedSize(60, 30)
                    delete_btn.setFixedSize(60, 30)

                    # 👉 버튼 기능 연결
                    edit_btn.clicked.connect(lambda _, lid=new_id: self.edit_line_description(lid))
                    delete_btn.clicked.connect(lambda _, lid=new_id: self.delete_line(lid))

                    # 👉 수평으로 묶기
                    line_widget = QWidget()
                    line_layout = QHBoxLayout()
                    line_layout.addWidget(label)
                    line_layout.addWidget(edit_btn)
                    line_layout.addWidget(delete_btn)
                    line_layout.setContentsMargins(0, 0, 0, 0)
                    line_widget.setLayout(line_layout)

                    self.right_layout.addWidget(line_widget)
                    self.line_labels[new_id] = label
                    self.line_widgets[new_id] = line_widget

                    # 👉 라벨에 단어 줄바꿈 설정
                    label.setWordWrap(True)

                    self.line_number += 1
                    self.max_line_number = max(self.max_line_number, self.line_number - 1)

                    # ✅ 누적 최대값 갱신
                    self.global_max_line_number = max(self.global_max_line_number, self.max_line_number)
    
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

                    area_id = len(self.stop_polygons)
                    label = QLabel(f"영역 {area_id} ({description})")
                    label.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
                    label.setWordWrap(True) 

                    edit_btn = QPushButton("수정")
                    delete_btn = QPushButton("삭제")
                    edit_btn.setFixedSize(60, 30)
                    delete_btn.setFixedSize(60, 30)

                    # 기능 연결
                    edit_btn.clicked.connect(lambda _, aid=area_id: self.edit_area_description(aid))
                    delete_btn.clicked.connect(lambda _, aid=area_id: self.delete_area(aid))

                    area_widget = QWidget()
                    layout = QHBoxLayout()
                    layout.addWidget(label)
                    layout.addWidget(edit_btn)
                    layout.addWidget(delete_btn)
                    layout.setContentsMargins(0, 0, 0, 0)
                    area_widget.setLayout(layout)

                    self.right_layout.addWidget(area_widget)
                    self.area_labels[area_id] = label
                    self.area_widgets[area_id] = area_widget

                    # self.stop_polygons.append(area_obj)
                    self.max_area_number = max(self.max_area_number, len(self.stop_polygons))
                    # ✅ 누적 최대값 갱신
                    self.global_max_area_number = max(self.global_max_area_number, self.max_area_number)
                    # self.area_number += 1

                    # self.undo_stack.append(("area", area_obj))  # 🔥 추가!
                    print(f"🚧 정지 감지 영역 {len(self.stop_polygons)} 생성 완료")
                else:
                    print("설명이 입력되지 않아 영역이 생성되지 않습니다.")
                    self.temp_points.clear()
                    self.update_display_with_lines()
                
                self.temp_points.clear()
            self.update_display_with_lines()

    def edit_line_description(self, line_id):
        old_desc = self.get_line_description(line_id)
        text, ok = QInputDialog.getText(self, f"선 {line_id} 설명 수정", "새 설명을 입력하세요:", text=old_desc)
        if ok and text.strip():
            new_desc = text.strip()
            for idx, (p1, p2, num, desc) in enumerate(self.lines):
                if num == line_id:
                    self.lines[idx] = (p1, p2, num, new_desc)
                    break
            self.line_labels[line_id].setText(f"선 {line_id} ({new_desc}): Count: {self.line_counts.get(line_id, 0)}")
            self.update_display_with_lines()

    def delete_line(self, line_id):
        self.lines = [line for line in self.lines if line[2] != line_id]
        if line_id in self.line_labels:
            self.line_labels[line_id].deleteLater()
            del self.line_labels[line_id]
        if line_id in self.line_widgets:
            self.right_layout.removeWidget(self.line_widgets[line_id])
            self.line_widgets[line_id].deleteLater()
            del self.line_widgets[line_id]
        self.update_display_with_lines()

    def edit_area_description(self, area_id):
        if area_id - 1 < len(self.stop_polygons):
            polygon, old_desc = self.stop_polygons[area_id - 1]
            text, ok = QInputDialog.getText(self, f"영역 {area_id} 설명 수정", "새 설명을 입력하세요:", text=old_desc)
            if ok and text.strip():
                new_desc = text.strip()
                self.stop_polygons[area_id - 1] = (polygon, new_desc)
                self.area_labels[area_id].setText(f"영역 {area_id} ({new_desc})")
                self.update_display_with_lines()

    def delete_area(self, area_id):
        if area_id - 1 < len(self.stop_polygons):
            del self.stop_polygons[area_id - 1]

        if area_id in self.area_labels:
            self.area_labels[area_id].deleteLater()
            del self.area_labels[area_id]

        if area_id in self.area_widgets:
            self.right_layout.removeWidget(self.area_widgets[area_id])
            self.area_widgets[area_id].deleteLater()
            del self.area_widgets[area_id]

        self.update_display_with_lines()

    # def keyPressEvent(self, event: QKeyEvent):
    #     if event.key() == Qt.Key_Q:
    #         print("Q 키 눌림: 종료")
    #         self.close()
    #     elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
    #         if self.drawing_enabled:
    #             print("Enter 키 눌림: 영상 재생 시작")
    #             self.drawing_enabled = False
    #             self.is_paused = False
    #             self.timer.start(30)
    #         else: # ⏯ 재생 중일 때 Enter로 일시정지 / 재개
    #             self.is_paused = not self.is_paused
    #             print("⏸ 일시정지" if self.is_paused else "▶ 재생")

    #     elif event.key() == Qt.Key_R:
    #         print("🔁 R 키: 상태 초기화")

    #         self.lines.clear()
    #         self.temp_points.clear()
    #         self.stop_polygons = [] # 영역 여러개 저장
    #         self.line_counts.clear()
    #         self.crossed_lines.clear()
    #         self.stop_watch.clear()
    #         self.prev_positions.clear()
    #         self.line_number = 1
    #         self.drawing_enabled = True
    #         self.draw_mode = 'line'
    #         self.line_mode_button.setChecked(True)
    #         self.area_mode_button.setChecked(False)
    #         self.cross_log.clear()  # ⏬ 선 통과 상태 초기화


    #         # 👉 선/영역 관련 위젯들 제거
    #         for widget in self.line_widgets.values():
    #             self.right_layout.removeWidget(widget)
    #             widget.deleteLater()
    #         self.line_widgets.clear()

    #         for widget in self.area_widgets.values():
    #             self.right_layout.removeWidget(widget)
    #             widget.deleteLater()
    #         self.area_widgets.clear()

    #         self.line_labels.clear()
    #         self.area_labels.clear()
    #         self.area_number = 1

    #         self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #         self.frame_idx = 1
    #         self.frame_label.setText("프레임: 1")  # ✅ 프레임 라벨도 초기화
    #         self.frame_slider.setValue(1)  # ✅ 트랙바도 1로 초기화

    #         self.search_frame_input.clear()

    #         ret, frame = self.cap.read()
    #         if ret:
    #             self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         self.update_display_with_lines()
    #         self.timer.stop()


    #     elif event.key() == Qt.Key_A:
    #         if self.frame_idx <= 1:
    #             print("첫 프레임입니다.")
    #             return
    #         self.frame_idx -= 1
    #         frame = self.safe_seek(self.frame_idx)
    #         if frame is not None:
    #             self.frame = frame 
    #             self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             self.frame_label.setText(f"프레임: {self.frame_idx}")
    #             self.frame_slider.setValue(self.frame_idx)
    #             self.force_draw_objects = True
    #             self.update_frame()
    #             self.force_draw_objects = False
                    
    #     elif event.key() == Qt.Key_D:
    #         if self.frame_idx >= self.total_frames:
    #             print("마지막 프레임입니다.")
    #             return
    #         self.frame_idx += 1
    #         frame = self.safe_seek(self.frame_idx)
    #         if frame is not None:
    #             self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             self.frame_label.setText(f"프레임: {self.frame_idx}")
    #             self.frame_slider.setValue(self.frame_idx)
    #             self.force_draw_objects = True
    #             self.update_frame()
    #             self.force_draw_objects = False
                
    def handle_slider_moved(self):
        value = self.frame_slider.value()
        frame = self.safe_seek(value)
        if frame is not None:
            self.frame_idx = value
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_label.setText(f"프레임: {self.frame_idx}")  # ✅ 영상 내부 기준
            self.frame_slider.setValue(self.frame_idx)
            self.force_draw_objects = True
            self.update_frame()
            self.force_draw_objects = False

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

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
    video_paths, _ = QFileDialog.getOpenFileNames(
        None,
        "여러 영상 파일 선택",
        "./assets",
        "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
    )
    if not video_paths:
        QMessageBox.warning(None, "경고", "영상 파일을 선택하지 않으면 프로그램이 종료됩니다.")
        sys.exit()

    # ✅ 라벨 파일 선택
    label_paths, _ = QFileDialog.getOpenFileNames(
        None,
        "여러 텍스트 파일 선택",
        "./assets",
        "Text Files (*.txt);;All Files (*)"
    )
    if not label_paths:
        QMessageBox.warning(None, "경고", "라벨 파일을 선택하지 않으면 프로그램이 종료됩니다.")
        sys.exit()

    video_label_pairs = []
    for v_path in video_paths:
        base = os.path.splitext(os.path.basename(v_path))[0]
        for l_path in label_paths:
            if base in l_path:  # 이름 매칭
                video_label_pairs.append((v_path, l_path))
                break

    if not video_label_pairs:
        QMessageBox.warning(None, "경고", "매칭되는 영상-라벨 쌍이 없습니다.")
        sys.exit()

    # ✅ 창 열기
    window = VideoWindow(video_label_pairs)
    window.show()
    sys.exit(app.exec_())