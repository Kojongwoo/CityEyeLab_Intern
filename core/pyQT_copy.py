import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QTextEdit,
    QPushButton, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import QTimer, Qt, QPoint, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen


class VideoWindow(QWidget):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("PyQt")

        # 전체 창 크기 설정
        window_width = 1800
        window_height = 900
        self.setFixedSize(window_width, window_height)

        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

        # 영상 표시 QLabel
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self.handle_mouse_press

        # 영상 QLabel 크기 (창의 90%)
        video_width = int(window_width * 0.9)
        video_height = int(window_height * 0.9)
        self.video_label.setFixedSize(video_width, video_height)

        # 오른쪽 입력 패널
        self.right_panel = QWidget(self)
        self.right_layout = QVBoxLayout()

        self.line_id_input = QLineEdit()
        self.line_id_input.setPlaceholderText("선 ID")

        self.line_desc_input = QTextEdit()
        self.line_desc_input.setPlaceholderText("선에 대한 설명 입력")

        self.apply_button = QPushButton("적용")  # 기능은 아직 없음

        self.right_layout.addWidget(self.line_id_input)
        self.right_layout.addWidget(self.line_desc_input)
        self.right_layout.addWidget(self.apply_button)
        self.right_panel.setLayout(self.right_layout)

        # 닫기 버튼
        self.close_button = QPushButton("닫기", self)
        self.close_button.clicked.connect(self.close)

        # 레이아웃 구성
        hbox = QHBoxLayout()
        hbox.addWidget(self.video_label, 8)
        hbox.addWidget(self.right_panel, 2)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.close_button)

        self.setLayout(vbox)

        # 영상 타이머 (처음엔 멈춤)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 선 관련 변수
        self.drawing_enabled = True
        self.temp_points = []
        self.lines = []

        self.show_first_frame()

        # 전역 키 입력 감지용 이벤트 필터
        self.installEventFilter(self)

    def show_first_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
            label_pos = event.pos()
            label_width = self.video_label.width()
            label_height = self.video_label.height()
            frame_height, frame_width, _ = self.frame.shape

            scale_x = frame_width / label_width
            scale_y = frame_height / label_height

            corrected_x = int(label_pos.x() * scale_x)
            corrected_y = int(label_pos.y() * scale_y)
            corrected_point = QPoint(corrected_x, corrected_y)

            self.temp_points.append(corrected_point)
            if len(self.temp_points) == 2:
                self.lines.append((self.temp_points[0], self.temp_points[1]))
                self.temp_points = []

            self.update_display_with_lines()

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress and self.drawing_enabled:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                print("Enter 감지됨 – 영상 재생 시작")
                self.drawing_enabled = False
                self.timer.start(30)
                return True  # 이벤트 처리 완료
        return super().eventFilter(source, event)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_path = './assets/2024-10-21 08_56_19.337.mp4'
    window = VideoWindow(video_path)
    window.show()
    sys.exit(app.exec_())
