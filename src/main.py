"""
Module Name: Ballistic detection

Description: TUP战队弹道散布量化测试工具

Author: YueYuanhaoo

Email: yueyuanhaoo@gmail.com

"""
import numpy as np
import time
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib.image as mping
import math
from sklearn.cluster import KMeans


# 二维码大小 41mm  装甲板大小 123  3X

camera_matrix = np.array([[1000, 0, 960], 
                          [0, 1000, 640], 
                          [0, 0, 1]], dtype=float)
# 如果没有畸变，也可以使用全零数组
dist_coeffs = np.zeros((5, 1))  


#读取图片
frame=cv2.imread('artest/ar_test4.jpg')
#调整图片大小
frame=cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)


#图片大小
size=frame.shape
h=size[0]
w=size[1]

#灰度化
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

origin_img=frame.copy()
#设置预定义的字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
#使用默认值初始化检测器参数
parameters =  aruco.DetectorParameters()



#使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
#画出标志位置
aruco.drawDetectedMarkers(frame, corners,ids)

center_point=np.array([w/2,h/2])
center_corners=np.mean(corners[0],axis=1)
#中心差值
dev_corners=corners[0]-center_corners
#中心放大选区
large_zone=center_point+4*dev_corners
large_zone_int=large_zone.astype(np.int32)
#放大选区可视化
cv2.polylines(frame,[large_zone_int],isClosed=True,color=(0,255,0),thickness=2)

# 空白可视化
zone_size=1600
pts=np.float32([[0,0],[0,zone_size],[zone_size,zone_size],[zone_size,0]])
large_zone_fp=large_zone.astype(np.float32)
marge=cv2.getPerspectiveTransform(large_zone_fp,pts)
dst=cv2.warpPerspective(origin_img,marge,(zone_size,zone_size))

dst_gray=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
ret1,th1 = cv2.threshold(dst_gray,195,255,cv2.THRESH_BINARY)


if ids is not None:
    # 估计姿态 (假设每个标记边长为0.05米)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
    # 旋转矩阵 平移矩阵 计算内部的标记角点
    #print(rvecs)
    # 绘制检测到的标记和位姿
    for i in range(len(ids)):
        img=cv2.aruco.drawDetectedMarkers(frame, corners)  # 绘制标记的边框
        img=cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)  # 绘制坐标轴


# 轮廓检测
contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
dst_find=cv2.drawContours(dst,contours,-1,(0,255,0),2)

# 设置SimpleBlobDetector参数
params = cv2.SimpleBlobDetector_Params()

# 改变阈值
# 只会检测minThreshold 和 maxThreshold之间的
params.minThreshold = 100
params.maxThreshold = 240

# 按Color：首先，您需要设置filterByColor = 1。
# 设置blobColor = 0来选择较暗的Blobs，设置blobColor = 255来选择较亮的Blobs。
params.filterByColor = True
params.blobColor = 0

# 根据面积过滤
# 按大小:可以根据大小过滤Blobs，方法是设置参数filterByArea = 1，以及适当的minArea和maxArea值。
# 例如，设置minArea = 100将过滤掉所有像素个数小于100的Blobs。
params.filterByArea = True
params.minArea = 400
params.maxArea = 15000

# 根据Circularity过滤，这个参数是(圆度)
# 这只是测量了这个blob与圆的距离。正六边形的圆度比正方形高。
# 要根据圆度进行过滤，设置filterByCircularity = 1。然后设置适当的minCircularity和maxCircularity值。
params.filterByCircularity = True
params.minCircularity = 0.1

# 根据Convexity过滤，这个参数是(凹凸性)
# 凸性定义为(Blob的面积/它的凸包的面积)。现在，凸包的形状是最紧的凸形状，完全包围了形状。
# 设置filterByConvexity = 1，然后设置0≤minConvexity≤1和maxConvexity(≤1)。
params.filterByConvexity = True
params.minConvexity = 0.14
params.maxConvexity = 1

# 根据Inertia过滤,惯性比
# 它衡量的是一个形状的伸长程度。例如，对于圆，这个值是1，对于椭圆，它在0和1之间，对于直线，它是0。
# 初步可以认为是外接矩形的长宽比，圆的外接矩形的长宽相等，椭圆是有长短轴，短轴长度除以长轴长度，介于0~1
# 直线可以认为没有宽度，因此是0
params.filterByInertia = True
params.minInertiaRatio = 0.1

# 创建筛选器
detector = cv2.SimpleBlobDetector_create(params)

# 检测blobs，得到一堆的关键点，之前我们在SIFT特征检测中也是得到了这个信息。
keypoints = detector.detect(th1)
keypoints_array = np.array(keypoints)

keypoint_x=[]
keypoint_y=[]
keypoint_size=[]
keypoint_xys=[]

for kp in keypoints:
    # print("关键点的位置是: pt[0]:{}\tpt[1]:{}\tangle:{}\tsize:{}".format(
    #     kp.pt[0], kp.pt[1], kp.angle, kp.size))
    keypoint_x.append(kp.pt[0])
    keypoint_y.append(kp.pt[1])
    keypoint_size.append(kp.size)

# 检测到关键点总个数
keypoint_nums=keypoints_array.shape
# print(keypoints_array.shape)

# 用红色圆圈画出检测到的blobs
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 确保圆的大小对应于blob的大小
im_with_keypoints = cv2.drawKeypoints(th1, keypoints_array,
                                      np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

blank_zone=np.zeros((zone_size,zone_size),np.uint8)
blank_zone.fill(255)

bold_point=cv2.drawKeypoints(blank_zone, keypoints_array,
                                      np.array([]), (255, 0, 0),
                                      cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
bold_point = cv2.drawKeypoints(bold_point, keypoints_array,
                                      np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 标记异常斑点
error_bold_x=[]
error_bold_y=[]
size_means=sum(keypoint_size)/len(keypoint_size)
size_means_bias=int(size_means)+2
size_radius=(size_means_bias/2)
# print(size_means_bias)
for bs in keypoint_size:
    if bs>size_means_bias:
        error_bold_x.append(keypoint_x[keypoint_size.index(bs)])
        error_bold_y.append(keypoint_y[keypoint_size.index(bs)])


# # 画出异常区域
for i in range(len(error_bold_x)):
    cv2.circle(im_with_keypoints, (int(error_bold_x[i]), int(error_bold_y[i])), int(size_radius), (0,255,0), 2)       


# 异常修复 拆分大圆为两个小圆/对ROI进行膨胀修复


# 数据整合
keypoint_ndarray=np.column_stack((keypoint_x,keypoint_y))

# 开始递归
kmeans=KMeans(n_clusters=1,n_init=10)
kmeans.fit(keypoint_ndarray)
y_kmeans=kmeans.predict(keypoint_ndarray)

# 聚类中心
center=kmeans.cluster_centers_
center_int_array=np.array([int(center[0][0]),int(center[0][1])])

cv2.circle(im_with_keypoints, center_int_array, 10, (0,255,0), -1)       
cv2.circle(bold_point, center_int_array , 10, (0,255,0), -1)       
bold_point_2=bold_point.copy()
cv2.putText(bold_point,str(keypoint_nums[0]),(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3,cv2.LINE_AA)

# 聚类连线 
# 距离列表
distance=[]
for i in range(len(keypoint_x)):
    keypoint_int_array=np.array([int(keypoint_x[i]),int(keypoint_y[i])])
    cv2.line(bold_point,center_int_array,keypoint_int_array,(1,1,1),1)
    #标记各点数字
    cv2.putText(bold_point,str(i+1),keypoint_int_array,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    # 计算各点距离中心的距离
    distance.append(np.linalg.norm(center_int_array-keypoint_int_array))



print("识别弹丸个数：",keypoint_nums[0])
# 统计圆分布
distance_means=int(sum(distance)/len(distance))
print("平均命中半径：",distance_means)

cv2.circle(bold_point_2, center_int_array , distance_means, (0,255,0), 2)     

# 最大框
# cv2.circle(bold_point_2, center_int_array , int(max(distance)), (255,0,0), 2)       

# 装甲板框大小
borad_size=600

cv2.rectangle(bold_point_2,center_int_array-borad_size,center_int_array+borad_size,(0,0,255),2)

# 装甲命中率
hit_rate=0
for dis in distance:
    if dis<borad_size:
        hit_rate=hit_rate+1

hit_rate=hit_rate/int(keypoint_nums[0])
hit_rate=format(hit_rate,'.4f')
print("装甲板最大命中率：",hit_rate)
cv2.putText(bold_point_2,str(hit_rate),(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3,cv2.LINE_AA)





# 将聚类中心作为新图像的中心点

# 前期效果图
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
key_rgb=cv2.cvtColor(im_with_keypoints,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.subplot(1,2,2)
plt.imshow(key_rgb)

# matplotlib 展示效果图
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Projectile detection')
plt.imshow(bold_point)

plt.subplot(1,2,2)
plt.title('Armor plate maximum hit')
plt.imshow(bold_point_2)


# matplotlib 统计数据：横纵直方图 
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(keypoint_x, bins=20)
plt.title('X-axis ballistic dispersion')
plt.xlabel('Distance')
plt.ylabel('Num')


plt.subplot(1,2,2)
plt.hist(keypoint_y, bins=20,orientation = 'horizontal')
plt.title('Y-axis ballistic dispersion')
plt.xlabel('Distance')
plt.ylabel('Num')

plt.show()




# 按键退出
# cv2.imshow("img",img)
# cv2.imshow("dst",dst_gray)
# cv2.imshow("th1",th1)
# cv2.imshow("blank_zone",bold_point)
# cv2.imshow("key",im_with_keypoints)
# cv2.imshow("blank_zone2",bold_point_2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

