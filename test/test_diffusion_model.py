import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 设置二维高斯分布的均值和协方差矩阵
mu = [0, 0]
Sigma = [[1, 0.5], [0.5, 1]]

# 生成二维网格
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# 计算二维高斯分布的概率密度函数值
rv = multivariate_normal(mu, Sigma)
Z = rv.pdf(pos)

# 计算 score function（对数概率密度函数对 x 的梯度）
log_pdf = np.log(Z)
Zx, Zy = np.gradient(log_pdf, x[1] - x[0], y[1] - y[0])

# 绘制等高线图
plt.figure(figsize=(8, 6))
contours = plt.contour(X, Y, log_pdf, levels=10, cmap="viridis")
plt.clabel(contours, inline=True, fontsize=8)

# 获取等高线的坐标点，并在这些点上绘制箭头
for collection in contours.collections:
    paths = collection.get_paths()
    for path in paths:
        # 获取等高线上的点
        vertices = path.vertices
        x_points = vertices[:, 0]
        y_points = vertices[:, 1]

        # 插值计算等高线上的 score function（梯度）值
        Zx_interp = np.interp(x_points, x, Zx[:, int(len(x) / 2)])
        Zy_interp = np.interp(y_points, y, Zy[int(len(y) / 2), :])

        # 绘制等高线上的箭头
        plt.quiver(x_points[::5], y_points[::5], Zx_interp[::5], Zy_interp[::5], 
                   color='red', scale=100)

plt.title("2D Gaussian Distribution with Score Function on Contours")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
