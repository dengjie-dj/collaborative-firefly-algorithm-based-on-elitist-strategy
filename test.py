import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import pairwise_distances_argmin_min
# 载入Iris数据集
filename = "synthetic_control.data"
X = pd.read_csv(filename, sep='\s+', header=None) # 如果文件有列标题则去掉 header=None
num_dimensions = X[1]
# iris = datasets.load_iris()
# X = iris.data
# 萤火虫算法参数
n_fireflies = 3  # 萤火虫数量
n_clusters = 3    # 聚类数量
max_iter = 50     # 最大迭代次数
alpha = 0.25      # 步长因子
gamma = 1.0       # 光吸收系数
beta_0 = 1.0      # 吸引度的基础值
# 初始化萤火虫位置（每个萤火虫表示一组聚类中心）
fireflies = np.random.rand(n_fireflies, n_clusters, X.shape[1]) * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
print(fireflies.shape)
print(X.shape)
print(fireflies)
# 目标函数（计算聚类内距离的总和）
def objective_function(X, centers):
    _, dist = pairwise_distances_argmin_min(X, centers)
    return np.sum(dist)
# 萤火虫亮度评估（目标函数值越小，亮度越高）
brightness = np.array([objective_function(X, firefly) for firefly in fireflies])
# 萤火虫算法主循环
for _ in range(max_iter):
    for i in range(n_fireflies):
        for j in range(n_fireflies):
            if brightness[i] > brightness[j]:  # i的亮度比j低，意味着i的目标函数值更高
                # 计算吸引度
                r = np.linalg.norm(fireflies[i] - fireflies[j])
                beta = beta_0 * np.exp(-gamma * r ** 2)
                # 向更亮的萤火虫移动
                fireflies[i] += beta * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(n_clusters, X.shape[1]) - 0.5)
                # 重新评估亮度
                brightness[i] = objective_function(X, fireflies[i])
                print(brightness[i])
# 选择最亮的萤火虫（目标函数值最小）作为最优解
best_firefly_index = np.argmin(brightness)
best_centers = fireflies[best_firefly_index]
# 根据最优聚类中心对数据进行聚类
cluster_labels, _ = pairwise_distances_argmin_min(X, best_centers)
# 输出聚类结果
print("Cluster assignments:", cluster_labels)