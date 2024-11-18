import cv2
import numpy as np

# 假设 price 是一个包含价格数据的数组
price = np.array([100, 110, 120, 130, 120, 110, 100, 90, 100, 110, 120, 130, 140, 150, 160, 150, 140, 130, 120, 110, 100])

# 使用 approxPolyDP 进行多边形逼近
epsilon = 0.01
approx = cv2.approxPolyDP(price.reshape((-1, 1)), epsilon, True)

# 打印逼近后的多边形顶点
print(approx)