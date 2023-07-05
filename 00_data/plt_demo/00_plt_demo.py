import matplotlib.pyplot as plt
import numpy as np


# 生成绘图数据部分，跳过即可
N = 21
x = np.linspace(0, 10, 11)
y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1,  9.9, 13.9, 15.1, 12.5]
a, b = np.polyfit(x, y, deg=1)
y_est = a * x + b
y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))

#########绘图开始####################################3

# 建立figure
fig = plt.figure(figsize=(12,8))

# 建立axes
ax = fig.add_subplot(111)

ax.plot(x, y_est, '-')  # 绘制直线
ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)  # 绘制区域
ax.plot(x, y, 'o', color='r')   # 绘制点集

# 添加属性
ax.set_xlabel('x axis',fontsize =16)
ax.set_ylabel('y axis',fontsize =16)
ax.set_title('example',loc='left')

ax.spines['top'].set_visible(False)  # 顶部坐标轴显示设置
ax.spines['right'].set_visible(False)  # 右部坐标轴显示设置

ax.set_xlim(0,20)
ax.set_yticks([0,5,10])
ax.set_yticklabels(['zero','five','ten'])

plt.show()