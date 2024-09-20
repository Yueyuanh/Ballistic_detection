import cv2
import numpy as np
import os

# 创建目录
os.makedirs('armark', exist_ok=True)

# DICT_4X4_50 
# Python: cv.aruco.DICT_4X4_50
# DICT_4X4_100 
# Python: cv.aruco.DICT_4X4_100
# DICT_4X4_250 
# Python: cv.aruco.DICT_4X4_250
# DICT_4X4_1000 
# Python: cv.aruco.DICT_4X4_1000
# DICT_5X5_50 
# Python: cv.aruco.DICT_5X5_50
# DICT_5X5_100 
# Python: cv.aruco.DICT_5X5_100
# DICT_5X5_250 
# Python: cv.aruco.DICT_5X5_250
# DICT_5X5_1000 
# Python: cv.aruco.DICT_5X5_1000
# DICT_6X6_50 
# Python: cv.aruco.DICT_6X6_50
# DICT_6X6_100 
# Python: cv.aruco.DICT_6X6_100
# DICT_6X6_250 
# Python: cv.aruco.DICT_6X6_250
# DICT_6X6_1000 
# Python: cv.aruco.DICT_6X6_1000
# DICT_7X7_50 
# Python: cv.aruco.DICT_7X7_50
# DICT_7X7_100 
# Python: cv.aruco.DICT_7X7_100
# DICT_7X7_250 
# Python: cv.aruco.DICT_7X7_250
# DICT_7X7_1000 
# Python: cv.aruco.DICT_7X7_1000
# DICT_ARUCO_ORIGINAL 
# Python: cv.aruco.DICT_ARUCO_ORIGINAL
# DICT_APRILTAG_16h5 
# Python: cv.aruco.DICT_APRILTAG_16h5
# 4x4 bits, minimum hamming distance between any two codes = 5, 30 codes

# 加载aruco字典
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# 生成标记
for i in range(30):
    markerImage = np.zeros((200, 200), dtype=np.uint8)
    markerImage = cv2.aruco.generateImageMarker(dictionary, i, 200)
    
    filename = 'armark/' + str(i) + '.png'
    cv2.imwrite(filename, markerImage)
