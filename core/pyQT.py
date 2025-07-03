import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel,
    QHBoxLayout, QVBoxLayout, QPushButton
)
from PyQt5.QtCore import QTimer, Qt, QPoint, QEvent
from PyQt5.QtGui import QImage, QPixmap, QKeyEvent, QPainter, QPen
from PyQt5.QtWidgets import QLineEdit, QTextEdit

class VideoWindow(QWidget):

    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("PyQt")
        # self.setGeometry(100, 100, 1000, 600)  # 넓은 창
        # ✅ 1. 전체 PyQt 창 크기 고정
        window_width = 1800
        window_height = 900
        # self.setFixedSize(window_width, window_height)


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

        # 선 ID 입력창
        self.line_id_input = QLineEdit()
        self.line_id_input.setPlaceholderText("선 ID")

        # 설명 입력창
        self.line_desc_input = QTextEdit()
        self.line_desc_input.setPlaceholderText("선에 대한 설명 입력")

        # 적용 버튼
        self.apply_button = QPushButton("적용")
        # (아직 기능은 연결하지 않음)

        # 입력창들 추가
        self.right_layout.addWidget(self.line_id_input)
        self.right_layout.addWidget(self.line_desc_input)
        self.right_layout.addWidget(self.apply_button)
        self.right_panel.setLayout(self.right_layout)

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
        self.lines = []        # [(QPoint, QPoint), ...] 형태로 선 저장        

        # ⏯ 영상 첫 프레임 미리 표시
        self.show_first_frame()

        # self.showMaximized()
        self.installEventFilter(self)


    def show_first_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 첫 프레임으로 되돌리기
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.update_display_with_lines()

    def update_frame(self):
        if not self.drawing_enabled:
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                self.cap.release()
                return

            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.update_display_with_lines()


    def update_display_with_lines(self):
        h, w, ch = self.frame.shape
        bytes_per_line = ch * w
        qimg = QImage(self.frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # 선 그리기
        painter = QPainter(pixmap)
        pen = QPen(Qt.red, 3)
        painter.setPen(pen)

        for p1, p2 in self.lines:
            painter.drawLine(p1, p2)
        for pt in self.temp_points:
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
            if len(self.temp_points) == 2:
                self.lines.append((self.temp_points[0], self.temp_points[1]))
                self.temp_points = []

            self.update_display_with_lines()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Q:
            print("Q 키 눌림 – 종료")
            self.close()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.drawing_enabled:
                print("Enter 키 눌림 – 영상 재생 시작")
                self.drawing_enabled = False
                self.timer.start(30)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_path = './assets/2024-10-21 08_56_19.337.mp4'  # 영상 경로 수정
    window = VideoWindow(video_path)
    window.show()
    sys.exit(app.exec_())