import sys, cv2, os, copy, json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, 
    QHBoxLayout, QVBoxLayout, QPushButton, QInputDialog,
    QFileDialog, QMessageBox,  QSlider, QLineEdit, QScrollArea, QComboBox
)
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QFont, QBrush, QColor
from datetime import datetime, timedelta
from utils import point_in_polygon
from pyproj import Transformer

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
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck_s',
    8: 'pm'
}

# LABEL_NAMES = {
#     0: 'car',
#     1: 'truck',
#     2: 'bus',
#     3: 'motor'
# }

def get_location_folder_key(path):
    return os.path.basename(os.path.dirname(path))

def read_raw_data(path, frame_offset=0):
    frame_data = {}
    frame_to_time = {}

    min_frame = float('inf')
    max_frame = 0

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 8:
                try:
                    frame, obj_id, x1, y1, x2, y2, label = map(int, parts[1:8])
                except ValueError:
                    continue

                adjusted_frame = frame - min_frame + 1 + frame_offset

                if adjusted_frame not in frame_data:
                    frame_data[adjusted_frame] = []
                    frame_to_time[adjusted_frame] = parts[0]  # 📌 이거만 있으면 됨

                frame_data[adjusted_frame].append((obj_id, x1, y1, x2, y2, label, parts[0]))

                min_frame = min(min_frame, frame)
                max_frame = max(max_frame, frame)

    return frame_data, 1 + frame_offset, max_frame - min_frame + 1, frame_to_time
    
# GPS 및 픽셀 기준점 (make_json.py에서 가져온 값 그대로 사용)
gps_top_left = (37.40105982169699,127.11294216334416)
gps_top_right = (37.40109597434296,127.11282504155552)
gps_bottom_left = (37.40150924020716,127.11290613188024)
gps_bottom_right = (37.40151269314831,127.1128284898534)

px_top_left = (809,168)
px_top_right = (990,195)
px_bottom_left = (856,710)
px_bottom_right = (1313,721)

# Transformer 정의
transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
transformer_to_gps = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)

# UTM 좌표로 변환
utm_top_left = transformer_to_utm.transform(gps_top_left[1], gps_top_left[0])
utm_top_right = transformer_to_utm.transform(gps_top_right[1], gps_top_right[0])
utm_bottom_left = transformer_to_utm.transform(gps_bottom_left[1], gps_bottom_left[0])
utm_bottom_right = transformer_to_utm.transform(gps_bottom_right[1], gps_bottom_left[0])

# Homography 계산
src_pts = np.array([px_top_left, px_top_right, px_bottom_right, px_bottom_left], dtype=np.float32)
dst_pts = np.array([utm_top_left, utm_top_right, utm_bottom_right, utm_bottom_left], dtype=np.float32)
H, _ = cv2.findHomography(src_pts, dst_pts)

# 픽셀 좌표를 GPS 좌표로 변환하는 함수
def pixel_to_gps(x, y):
    pt = np.array([[x, y, 1]], dtype=np.float32).T
    result = H @ pt
    result /= result[2]
    utm_x, utm_y = result[0][0], result[1][0]
    lon, lat = transformer_to_gps.transform(utm_x, utm_y)
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
        self.setWindowTitle("Watching Tool_old")
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
        # self.video_label.mousePressEvent = self.handle_mouse_press  # ✅ 마우스 클릭 이벤트 등록
        
        # 영상 QLabel 크기 = 전체 창의 90%
        video_width = int(window_width * 0.8)
        video_height = int(window_height * 0.95)
        self.video_label.setFixedSize(video_width, video_height)

        # 우측 패널 레이아웃 조정
        self.right_scroll = QScrollArea()
        self.right_scroll.setFixedWidth(360)  # 너비 고정 (원하는 값)
        self.right_scroll.setWidgetResizable(True)
        
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_layout.setAlignment(Qt.AlignTop)  # 핵심: 위로 정렬
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(12)
        self.right_panel.setLayout(self.right_layout)

        self.right_scroll.setWidget(self.right_panel)

        # ✅ 콤보박스 위에 '영상 선택' 라벨 추가
        video_select_label = QLabel("🎞 영상 선택")
        video_select_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #444;")
        self.right_layout.addWidget(video_select_label)

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
        self.cumulative_frame_offset = 0  # 누적 프레임 오프셋

        self.frame_data, self.min_frame, self.max_frame, self.frame_time_map = read_raw_data(self.label_path, frame_offset=self.cumulative_frame_offset)
        self.frame_idx = self.min_frame  # 항상 1이 됨

        self.per_file_states = {}  # 각 영상별 상태 저장용 딕셔너리

        # 닫기 버튼
        self.close_button = QPushButton("닫기", self)
        self.close_button.clicked.connect(self.close)

        # 수평 레이아웃: 왼쪽 영상 + 오른쪽 빈 영역
        hbox = QHBoxLayout()
        hbox.setSpacing(20)  # ← 영상과 오른쪽 패널 사이 간격 설정
        hbox.addWidget(self.video_label, stretch = 0)  
        # hbox.addWidget(self.right_panel)  
        hbox.addWidget(self.right_scroll)  

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

        # 결과 저장용 csv 초기화
        video_date_str = os.path.basename(self.video_path).split()[0]  # "2024-10-21"
        today_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_csv = f"./logs/{video_date_str}_analyzed_{today_str}.csv"
        self.csv_header_written = False

        base_name = f"{video_date_str}_analyzed_{today_str}"
        csv_dir = "./logs"

        # 파일명 중복 방지: v2, v3, ...
        version = 1
        while True:
            if version == 1:
                csv_path = os.path.join(csv_dir, f"{base_name}_v{version}.csv")
            else:
                csv_path = os.path.join(csv_dir, f"{base_name}_v{version}.csv")
            if not os.path.exists(csv_path):
                break
            version += 1

        self.output_csv = csv_path
        self.csv_header_written = False

        # JSON 저장 경로
        self.json_path = "./logs/lines_areas.json"

        # ⏯ 영상 첫 프레임 미리 표시
        self.show_first_frame()

        self.stop_polygons = []  # → [ ([QPoint, QPoint, QPoint, QPoint], "설명"), ... ]

        self.installEventFilter(self)

        # 선 통과 여부 저장용 딕셔너리 추가
        self.cross_log = set()  # (obj_id, line_id) → 통과 여부
        self.line_cross_once_logged = set()  # 👉 (obj_id, line_id) → 최초 기록 여부 추적
        self.area_cross_once_logged = set()  # 👉 (obj_id, area_id)

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

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ⏯ 영상 재생 / 일시정지 버튼
        self.play_pause_button = QPushButton("⏯ 재생 / 일시정지")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.right_layout.addWidget(self.play_pause_button)
        self.play_pause_button.setFixedHeight(40)

        # 📂 이전 / 다음 영상 버튼
        self.prev_video_button = QPushButton("📂 이전 영상")
        self.next_video_button = QPushButton("📂 다음 영상")
        self.prev_video_button.setFixedHeight(40)
        self.next_video_button.setFixedHeight(40)

        # 버튼 클릭 연결
        self.prev_video_button.clicked.connect(self.go_prev_video)
        self.next_video_button.clicked.connect(self.go_next_video)

        # 한 줄로 정렬
        video_nav_layout = QHBoxLayout()
        video_nav_layout.addWidget(self.prev_video_button)
        video_nav_layout.addWidget(self.next_video_button)
        self.right_layout.addLayout(video_nav_layout)

        # 🔁 초기화 버튼
        self.reset_button = QPushButton("🔁 처음으로 돌아가기")
        self.reset_button.clicked.connect(self.reset_video_state)
        self.right_layout.addWidget(self.reset_button)
        self.reset_button.setFixedHeight(40)

        # 실시간 라벨
        self.time_label = QLabel("시간: -")
        self.time_label.setStyleSheet("color: darkgreen; font-size: 20px;")
        self.right_layout.addWidget(self.time_label)

        # ✅ 슬라이더 디바운싱용 QTimer 추가
        self.slider_preview_timer = QTimer()
        self.slider_preview_timer.setSingleShot(True)
        self.slider_preview_timer.timeout.connect(self.apply_slider_preview)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(self.total_frames)
        self.frame_slider.setValue(self.frame_idx)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setSingleStep(1)
        
        self.frame_slider.sliderReleased.connect(self.handle_slider_moved)
        self.frame_slider.sliderPressed.connect(self.pause_for_slider)

        # ✅ 디바운싱 처리용 valueChanged 연결
        self.frame_slider.valueChanged.connect(self.delayed_slider_preview)

        self.right_layout.addWidget(self.frame_slider)

        # 🔄 5초 되돌리기 / 앞으로 가기 버튼 추가
        self.skip_back_button = QPushButton("⏪ 5초 되돌리기")
        self.skip_forward_button = QPushButton("⏩ 5초 앞으로 가기")
        self.skip_back_button.setFixedHeight(40)
        self.skip_forward_button.setFixedHeight(40)

        # 👉 함수 연결
        self.skip_back_button.clicked.connect(lambda: self.skip_seconds(-5))
        self.skip_forward_button.clicked.connect(lambda: self.skip_seconds(5))

        # 👉 수평 정렬
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(self.skip_back_button)
        skip_layout.addWidget(self.skip_forward_button)
        self.right_layout.addLayout(skip_layout)
        

        self.force_draw_objects = False  # 👉 객체 박스 강제 그리기 용도
        self.group_states = {}  # 👉 폴더별(장소별) 상태 저장

    def go_prev_video(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.change_file(self.current_index)
        else:
            QMessageBox.information(self, "알림", "첫 번째 영상입니다.")

    def go_next_video(self):
        if self.current_index + 1 < len(self.video_label_pairs):
            self.current_index += 1
            self.change_file(self.current_index)
        else:
            QMessageBox.information(self, "알림", "마지막 영상입니다.")

    def toggle_play_pause(self):
        if self.drawing_enabled:
            self.drawing_enabled = False
            self.is_paused = False
            self.timer.start(30)
        else:
            self.is_paused = not self.is_paused

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
        self.cross_log.clear()
        self.area_number = 1

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 1
        self.time_label.setText(f"시간: {self.frame_time_map[self.frame_idx]}")
        self.frame_slider.setValue(1)
        # self.search_frame_input.clear()

        ret, frame = self.cap.read()
        if ret:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.update_display_with_lines()
        self.timer.stop()
        
    def change_file(self, index):
        # 👉 현재 상태 저장
        if 0 <= self.current_index < len(self.video_label_pairs):
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
        # ✅ 상태 초기화 (이전 영상 잔상 제거)
        self.frame_data = {}
        self.lines = []
        self.stop_polygons = []
        self.line_counts = {}
        self.crossed_lines = set()
        self.stop_watch = {}
        self.illegal_log = set()
        self.prev_positions = {}
        self.line_labels = {}
        self.area_labels = {}
        self.line_widgets = {}
        self.area_widgets = {}

        self.current_index = index

        self.video_path, self.label_path = self.video_label_pairs[index]

        # ✅ 오프셋 제거된 read_raw_data 호출
        # self.frame_data, self.min_frame, self.max_frame = read_raw_data(self.label_path)
        self.frame_data, self.min_frame, self.max_frame, self.frame_time_map = read_raw_data(self.label_path, frame_offset=self.cumulative_frame_offset)


        # ✅ frame_data 비었을 때 경고
        if not self.frame_data:
            print(f"[ERROR] frame_data is EMPTY from label: {self.label_path}")

        self.frame_idx = self.min_frame  # 객체가 있는 첫 프레임으로 정확히 설정

        # ✅ 여기 아래에 영상 이름 라벨 갱신 추가!
        self.video_name_label.setText(f"🎬 현재 영상: {os.path.basename(self.video_path)}")

        # 영상 재로딩
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_slider.setMaximum(self.total_frames)

        # 👉 새 영상 상태 복원 or 초기화
        state = self.per_file_states.get(self.video_path, None)

        if state:
            # self.frame_idx = state["frame_idx"]
            # ✅ frame_idx 유효성 검사
            if self.min_frame <= state["frame_idx"] <= self.max_frame:
                self.frame_idx = state["frame_idx"]
            else:
                print(f"[WARN] 복원된 frame_idx {state['frame_idx']} 가 범위를 벗어남. 초기화함.")
                self.frame_idx = self.min_frame

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
            self.frame_idx = self.min_frame
            self.line_number = 1
            self.area_number = 1

        # 👉 우측 라벨 및 위젯 초기화
        # self.clear_shape_widgets()

        # 👉 폴더별 그룹 상태 복원
        group_key = get_location_folder_key(self.video_path)
        group_state = self.group_states.get(group_key, None)
        if group_state:
            self.lines = copy.deepcopy(group_state["lines"])
            self.stop_polygons = copy.deepcopy(group_state["stop_polygons"])
            self.line_number = group_state["line_number"]
            self.area_number = group_state["area_number"]

        # 👉 선 위젯 추가
        sorted_lines = sorted(self.lines, key=lambda x: x[2])  # x[2] = line_id
        # 👉 선 다시 추가
        for p1, p2, line_id, desc in sorted_lines:
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

            self.scroll_layout.addWidget(widget)  # ✅ 스크롤 영역에 추가
            self.line_labels[line_id] = label
            self.line_widgets[line_id] = widget

        # ✅ 영역 번호 기준으로 정렬
        sorted_areas = sorted(enumerate(self.stop_polygons), key=lambda x: x[0] + 1)
        for idx, (polygon, desc) in sorted_areas:
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
            # self.right_layout.addWidget(widget)
            self.scroll_layout.addWidget(widget)
            self.area_labels[area_id] = label
            self.area_widgets[area_id] = widget

        # ✅ 현재 선/영역 리스트에서 최대 ID를 기준으로 line_number, area_number 재설정
        existing_line_ids = [line[2] for line in self.lines]  # line = (p1, p2, line_id, desc)
        self.line_number = max(existing_line_ids, default=0) + 1
        self.area_number = len(self.stop_polygons) + 1

        # 프레임 로딩 및 디스플레이
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx - 1)
        ret, frame = self.cap.read()

        if ret:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.update_display_with_lines()
            # self.frame_label.setText(f"프레임: {self.frame_idx}")
            self.frame_slider.setValue(self.frame_idx)

            # ✅ 시간 라벨 업데이트
            if self.frame_idx in self.frame_time_map:
                self.time_label.setText(f"시간: {self.frame_time_map[self.frame_idx]}")
            else:
                self.time_label.setText("시간: -")

        # ✅ 여기에 이 2줄 추가
        self.drawing_enabled = True
        self.is_paused = True

        # print(f"[DEBUG] 🔄 영상 전환됨: {os.path.basename(self.video_path)}")
        # print(f"[DEBUG] frame_data keys: {list(self.frame_data.keys())[:5]}")
        # print(f"[DEBUG] frame_idx = {self.frame_idx}")

    def get_line_description(self, line_id):
        for p1, p2, num, desc in self.lines:
            if num == line_id:
                return desc
        return ""
    
    def skip_seconds(self, seconds):
        # """현재 시간에서 ±seconds 후 가장 가까운 프레임으로 이동"""
        if self.frame_idx not in self.frame_time_map:
            # print("[ERROR] 현재 프레임 시간 정보 없음")
            return

        # 현재 시간 가져오기
        current_time_str = self.frame_time_map[self.frame_idx]
        try:
            current_time = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            print("[ERROR] 시간 파싱 실패:", current_time_str)
            return

        # 목표 시간 계산
        target_time = current_time + timedelta(seconds=seconds)

        # 가장 가까운 프레임 찾기
        target_frame = self.find_closest_frame_to_time(target_time)

        # ✅ 최소 프레임보다 작으면 첫 프레임으로 보정
        if target_frame < self.min_frame:
            print("[INFO] 영상 시작보다 앞이라 첫 프레임으로 이동")
            target_frame = self.min_frame

        # ✅ 최대 프레임보다 크면 마지막 프레임으로 보정
        if target_frame > self.total_frames:
            print("[INFO] 영상 끝보다 뒤라 마지막 프레임으로 이동")
            target_frame = self.total_frames

        # 이동 수행
        frame = self.safe_seek(target_frame)
        if frame is not None:
            self.frame_idx = target_frame
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.frame_idx in self.frame_time_map:
                self.time_label.setText(f"시간: {self.frame_time_map[self.frame_idx]}")
            else:
                self.time_label.setText("시간: -")

            self.frame_slider.setValue(self.frame_idx)
            self.force_draw_objects = True
            self.update_frame()
            self.force_draw_objects = False

    def find_closest_frame_to_time(self, target_time: datetime) -> int:
        closest_frame = self.frame_idx
        closest_diff = timedelta.max

        for f, time_str in self.frame_time_map.items():
            try:
                frame_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                continue

            diff = abs(frame_time - target_time)
            if diff < closest_diff:
                closest_diff = diff
                closest_frame = f

        return closest_frame
    
    def safe_seek(self, target_frame):
        # """빠르게 특정 프레임으로 이동"""
        # 1프레임부터 시작 → OpenCV는 0-based니까
        frame_idx = max(0, int(target_frame - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print(f"[ERROR] Frame {target_frame} read failed")
            return None

        return frame.copy()
    
    # def safe_seek(self, target_frame):
    #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #     for i in range(target_frame - 1):
    #         ret, _ = self.cap.read()
    #         if not ret:
    #             print(f"[ERROR] Frame {i+1} read failed during seek")

    #     ret, frame = self.cap.read()
    #     if not ret:
    #         print(f"[ERROR] Frame {target_frame} read failed at target")
    #         return None

    #     if frame is None:
    #         print(f"[ERROR] Frame {target_frame} is None")
    #         return None

    #     return frame.copy()  # ✅ 반드시 frame 반환해야 정상 동작

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
            for obj_id, x1, y1, x2, y2, label, timestamp_str in self.frame_data[self.frame_idx]:
                color = LABEL_COLORS.get(label, DEFAULT_COLOR)
                label_name = LABEL_NAMES.get(label, f"Label:{label}")

                # 바운딩 박스 및 테스트
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                # 객체 ID + 라벨명
                cv2.putText(frame_rgb, f"ID:{obj_id}, {label_name}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                # 중심 좌표 계산 및 시각화
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(frame_rgb, (cx, cy), 3, color, -1)

                # cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                curr_point = QPoint(cx, cy)

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

        # ✅ 시간 라벨만 갱신
        if self.frame_idx in self.frame_time_map:
            self.time_label.setText(f"시간: {self.frame_time_map[self.frame_idx]}")
        else:
            self.time_label.setText("시간: -")

        self.frame_slider.setValue(self.frame_idx)
                           
    def update_display_with_lines(self):
        # print(f"[DRAW] Displaying frame {self.frame_idx}")
        h, w, ch = self.frame.shape
        bytes_per_line = ch * w
        qimg = QImage(self.frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation
        ))
        
    def handle_slider_moved(self):
        value = self.frame_slider.value()
        frame = self.safe_seek(value)
        if frame is not None:
            self.frame_idx = value
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ✅ 시간 라벨만 갱신
            if self.frame_idx in self.frame_time_map:
                self.time_label.setText(f"시간: {self.frame_time_map[self.frame_idx]}")
            else:
                self.time_label.setText("시간: -")
                
            self.frame_slider.setValue(self.frame_idx)
            self.force_draw_objects = True
            self.update_frame()
            self.force_draw_objects = False

        if self.was_playing:
            self.timer.start(30)  # 30ms = 약 33fps

    def delayed_slider_preview(self, value):
        self.pending_slider_value = value
        self.slider_preview_timer.start(50)  # 50ms 지연 후 apply 호출

    def apply_slider_preview(self):
        value = self.pending_slider_value
        frame = self.safe_seek(value)
        if frame is not None:
            self.frame_idx = value
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.frame_idx in self.frame_time_map:
                self.time_label.setText(f"시간: {self.frame_time_map[self.frame_idx]}")
            else:
                self.time_label.setText("시간: -")

            self.frame_slider.setValue(self.frame_idx)
            self.force_draw_objects = True
            self.update_frame()
            self.force_draw_objects = False

    def pause_for_slider(self):
    # """슬라이더를 잡는 순간 재생 멈춤"""
        if self.timer.isActive():
            self.timer.stop()
            self.was_playing = True
        else:
            self.was_playing = False

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # ✅ 영상 파일 선택
    video_paths, _ = QFileDialog.getOpenFileNames(
        None,
        "여러 영상 파일 선택",
        "./K8",
        "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
    )
    if not video_paths:
        QMessageBox.warning(None, "경고", "영상 파일을 선택하지 않으면 프로그램이 종료됩니다.")
        sys.exit()

    # ✅ 라벨 파일 선택
    label_paths, _ = QFileDialog.getOpenFileNames(
        None,
        "여러 텍스트 파일 선택",
        "./K8",
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