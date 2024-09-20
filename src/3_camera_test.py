import numpy as np
import time
import cv2
import cv2.aruco as aruco

#相机参数，至少为1
camera_matrix = np.array([[1000, 0, 320], 
                          [0, 1000, 240], 
                          [0, 0, 1]], dtype=float)

# 如果没有畸变，也可以使用全零数组
dist_coeffs = np.zeros((5, 1))  

#设置预定义的字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

#打开摄像头
cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:

    #读取图片
    ret, frame=cap.read()

    #调整图片大小
    frame=cv2.resize(frame,None,fx=1,fy=1,interpolation=cv2.INTER_CUBIC)

    #图片大小
    size=frame.shape
    h=size[0]
    w=size[1]

    #灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #使用默认值初始化检测器参数
    parameters =  aruco.DetectorParameters()


    #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

    #画出标志位置
    aruco.drawDetectedMarkers(frame, corners,ids)

    # print(corners)


    if ids is not None:
        # 估计姿态 (假设每个标记边长为0.05米)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        # 旋转矩阵 平移矩阵 计算内部的标记角点


        # 绘制检测到的标记和位姿
        for i in range(len(ids)):
            frame=cv2.aruco.drawDetectedMarkers(frame, corners)  # 绘制标记的边框
            frame=cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)  # 绘制坐标轴
            
            # 输出位姿信息
            print(f"Marker ID: {ids[i][0]}")
            print(f"Rotation Vector (rvec): {rvecs[i].flatten()}")
            print(f"Translation Vector (tvec): {tvecs[i].flatten()}")

        center_point=np.array([w/2,h/2])
        center_corners=np.mean(corners[0],axis=1)
        #中心差值
        dev_corners=corners[0]-center_corners
        #中心放大选区
        large_zone=center_point+4*dev_corners
        large_zone_int=large_zone.astype(np.int32)
        #放大选区可视化
        cv2.polylines(frame,[large_zone_int],isClosed=True,color=(0,255,0),thickness=2)


        zone_size=800
        pts=np.float32([[0,0],[0,zone_size],[zone_size,zone_size],[zone_size,0]])
        large_zone_fp=large_zone.astype(np.float32)
        marge=cv2.getPerspectiveTransform(large_zone_fp,pts)
        dst=cv2.warpPerspective(gray,marge,(zone_size,zone_size))
        cv2.imshow('dst',dst)


    cv2.imshow("image",frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()