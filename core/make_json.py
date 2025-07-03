import csv
import json
from datetime import datetime, timedelta
import numpy as np
import cv2
from pyproj import Proj, Transformer

# 실제 GPS 및 픽셀 좌표들
gps_top_left = (37.40105982169699,127.11294216334416)
gps_top_right = (37.40109597434296,127.11282504155552)
gps_bottom_left = (37.40150924020716,127.11290613188024)
gps_bottom_right = (37.40151269314831,127.1128284898534)

px_top_left = (809,168)
px_top_right = (990,195)
px_bottom_left = (856,710)
px_bottom_right = (1313,721)

# UTM 변환기 초기화
transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
transformer_to_gps = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)

# GPS → UTM
utm_top_left = transformer_to_utm.transform(gps_top_left[1], gps_top_left[0])
utm_top_right = transformer_to_utm.transform(gps_top_right[1], gps_top_right[0])
utm_bottom_left = transformer_to_utm.transform(gps_bottom_left[1], gps_bottom_left[0])
utm_bottom_right = transformer_to_utm.transform(gps_bottom_right[1], gps_bottom_right[0])

# Homography 매핑
src_pts = np.array([px_top_left, px_top_right, px_bottom_right, px_bottom_left], dtype=np.float32)
dst_pts = np.array([utm_top_left, utm_top_right, utm_bottom_right, utm_bottom_left], dtype=np.float32)
H, _ = cv2.findHomography(src_pts, dst_pts)

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

def pixel_to_gps(x, y):
    # 반전 제거 - 영상 그대로 사용
    pt = np.array([[x, y, 1]], dtype=np.float32).T
    result = H @ pt
    result /= result[2]
    utm_x, utm_y = result[0][0], result[1][0]
    lon, lat = transformer_to_gps.transform(utm_x, utm_y)
    return lat, lon

# 라벨 매핑
label_map = {
    0: 'car',
    1: 'bus_s',
    2: 'bus_m',
    3: 'truck_s',
    4: 'truck_m',
    5: 'truck_x',
    6: 'bike'
}

csv_path = "C:/Work/Projects/Solution_DEV/CITY_TRAFFIC/Working/assets/2024-10-21 08_16_26.63.txt"
json_path = "output.json"

base_time = datetime.strptime("2024-10-21T08:12:45Z", "%Y-%m-%dT%H:%M:%SZ")
data_list = []

with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        frame_num = int(row[0])
        obj_id = int(row[1])
        x1 = float(row[2])
        y1 = float(row[3])
        x2 = float(row[4])
        y2 = float(row[5])
        label_val = int(row[6])

        # 바운딩 박스 중심 좌표
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 좌우/상하 반전 제거 후 바로 변환
        lat, lon = pixel_to_gps(center_x, center_y)
        altitude = 0
        label = label_map.get(label_val, "unknown")

        item = {
            "frame": frame_num,
            "id": obj_id,
            "gps": {
                "lat": center_x,
                "lng": center_y
            },
            "altitude": altitude,
            "label": label
        }
        data_list.append(item)

with open(json_path, "w") as f:
    json.dump(data_list, f, indent=2)

print(f"Saved {len(data_list)} records to {json_path}")
