# utils.py
import cv2
import numpy as np

def point_in_polygon(pt, polygon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))  # numpy 배열로 보정
    return cv2.pointPolygonTest(polygon, pt, False) >= 0
