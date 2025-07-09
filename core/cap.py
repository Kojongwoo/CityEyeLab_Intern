import cv2

cap = cv2.VideoCapture("./assets/2024-10-21 08_12_45.644.mp4")

for i in range(1, 300):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    print("총 프레임 수:", int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

cap.release()
