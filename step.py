import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
# 载入Iris数据集
filename = "synthetic_control.data"
X = pd.read_csv(filename, sep='\s+', header=None) # 如果文件有列标题则去掉 header=None

num_dimensions = X.shape[1]

print(num_dimensions)
# 除去最后一列
# X = X.iloc[:, :-1]
print(X.shape)
# 萤火虫算法参数
n_fireflies =400  # 萤火虫数量
n_clusters = 6    # 聚类数量
max_iter = 100    # 最大迭代次数
alpha = 0.25      # 步长因子
gamma = 1.0       # 光吸收系数
beta_0 = 1.0      # 吸引度的基础值
# 初始化萤火虫位置（每个萤火虫表示一组聚类中心）
bounds = (X.min(axis=0), X.max(axis=0))
fireflies = np.random.uniform(bounds[0], bounds[1], (n_fireflies, n_clusters, num_dimensions))

# 目标函数（计算聚类内距离的总和）
def objective_function(X, centers):
    _, dist = pairwise_distances_argmin_min(X, centers)
    return np.sum(dist)
# 萤火虫亮度评估（目标函数值越小，亮度越高）
brightness = np.array([objective_function(X, firefly) for firefly in fireflies])
# 萤火虫算法主循环
def fA_step():
    beta_alpha = 0.55 / max_iter
    alphas = np.full(n_fireflies, 0.5)  # 为每个萤火虫初始化一个随机的alpha值
    for it in range(max_iter):
        print(it)
        # 找到亮度最高的萤火虫
        brightest_firefly_index = np.argmax(brightness)
        for i in range(n_fireflies):
            # 为亮度最高的萤火虫采取特定的alpha计算策略
            if i == brightest_firefly_index:

                alphas[i] = alphas[i]+beta_alpha # 这里填写计算亮度最高的萤火虫的alpha值的策略
            else:

                alphas[i] = alphas[i]-beta_alpha # 这里填写计算其他萤火虫的alpha值的策略
            for j in range(n_fireflies):
                if brightness[i] > brightness[j]:  # i的亮度比j低，意味着i的目标函数值更高
                    # 计算吸引度
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta_0 * np.exp(-gamma * r ** 2)
                    # 向更亮的萤火虫移动
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + alphas[i] * (np.random.rand(n_clusters, X.shape[1]) - 0.5)
                    # 重新评估亮度
                    brightness[i] = objective_function(X, fireflies[i])
    brightest_firefly_fitness = brightness[brightest_firefly_index]
    return brightest_firefly_fitness
# 选择最亮的萤火虫（目标函数值最小）作为最优解
results = []
count = 0
max_value = float('-inf')  # 初始化为负无穷大
min_value = float('inf')   # 初始化为正无穷大
for _ in range(5):
    best_value = fA_step()
    results.append(best_value)
    if best_value < 0.000001:
        count += 1
    # 更新最大值和最小值
    if best_value > max_value:
        max_value = best_value
    if best_value < min_value:
        min_value = best_value
# 计算统计数据
mean_result = np.mean(results)
std_deviation = np.std(results)
probability_of_acceptable_error = count / 10  # 这里应该是除以10，因为循环是10次
print(f"平均值: {mean_result}")
print(f"标准差: {std_deviation}")
print(f"到达可接受误差范围的几率: {probability_of_acceptable_error}")
print(f"最大值: {max_value}")
print(f"最小值: {min_value}")