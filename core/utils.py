# utils.py
import cv2
import numpy as np

def point_in_polygon(pt, polygon):
    pts = np.array([[p.x(), p.y()] for p in polygon], dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(pts, pt, False) >= 0
