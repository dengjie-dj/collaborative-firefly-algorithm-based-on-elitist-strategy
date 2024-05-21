import math
import random
import numpy as np
from sklearn import datasets
from sklearn.metrics import pairwise_distances_argmin_min
# 载入Iris数据集
wine = datasets.load_wine()
X = wine.data
print(X.shape)
# 萤火虫算法参数

n_clusters = 3    # 聚类数量

# 初始化萤火虫位置（每个萤火虫表示一组聚类中心）


# 目标函数（计算聚类内距离的总和）
def objective_function(X, centers):
    _, dist = pairwise_distances_argmin_min(X, centers)
    return np.sum(dist)
# 萤火虫亮度评估（目标函数值越小，亮度越高）





# 初始化萤火虫群体


# 保存每一代的最佳适应度
my_best_fitness_history = []
convergence_count = 0
acceptable_error=0.000001
def generate_new_solution_by_swapping_features(current_solution):
    new_solution = current_solution[:]
    # 随机选择两个特征进行交换
    i, j = random.sample(range(len(new_solution)), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution
# 萤火虫算法主循环
def myFA():
# 萤火虫算法参数
    num_fireflies = 400

    alpha = 0.25  # 步长参数
    gamma = 1.0  # 光吸收系数
    beta_0 = 1.0  # 最大吸引力
    # num_elites = 5
    temp = 10
    cooling_rate = 0.9
    num_elites = num_fireflies // 6  # 确保是整数  # 精英数量
    # 初始化萤火虫群体



    num_dimensions = X.shape[1]
    max_iter = 200
    bounds = (X.min(axis=0), X.max(axis=0))
    fireflies = np.random.uniform(bounds[0], bounds[1], (num_fireflies, n_clusters, num_dimensions))

    light_intensity = np.array([objective_function(X, firefly) for firefly in fireflies])
    light_intensity = np.array([objective_function(X,f) for f in fireflies])
    # 保存每一代的最佳适应度
    my_best_fitness_history = []
    convergence_count = 0
    acceptable_error=0.000001

    # 萤火虫算法主循环
    for iter in range(max_iter):
        # 对萤火虫进行排序并找到最亮的萤火虫
        sorted_indices = np.argsort(light_intensity)
        fireflies = fireflies[sorted_indices]
        light_intensity = light_intensity[sorted_indices]

        # 分离精英群体和非精英群体
        elite_fireflies = fireflies[:num_elites]
        non_elite_fireflies = fireflies[num_elites:]
        elite_intensity = light_intensity[:num_elites]

        masses = 1/elite_intensity
        # 计算质量的总和
        total_mass = np.sum(masses)
        # 计算每个维度的重心位置
        centroid = np.zeros((3,num_dimensions))
        print(elite_fireflies[1].shape)
        print(masses[1].shape)
        print(centroid.shape)
        for i in range(num_elites):
            centroid += elite_fireflies[i] * masses[i]
        centroid /= total_mass

        non_elite_intensity = light_intensity[num_elites:]

        ga = (max_iter-iter+1)/max_iter

        sjian = 0.9

        alpha = sjian*alpha
        betamin = 0.2
        # 精英群体内部迭代
        for i in range(num_elites):
            # for j in range(num_elites):
            range_list = list(range(num_elites))
            range_list.remove(i)
            j = random.sample(range_list, 1)[0]
            if elite_intensity[i] > elite_intensity[j]:  # i向j移动
                # elite_fireflies[i] = (1 - ga) * elite_fireflies[i] +  ga * elite_fireflies[j] + alpha * (np.random.rand(num_dimensions) - 0.5)
                elite_fireflies[i] = ga*elite_fireflies[i]+(1-ga)*elite_fireflies[j]+alpha*(np.random.rand(num_dimensions) - 0.5)
                    # r = np.linalg.norm(elite_fireflies[i] - elite_fireflies[j])
                    # beta = beta_0 * np.exp(-gamma * r ** 2)
                    # elite_fireflies[i] += beta * (elite_fireflies[j] - elite_fireflies[i]) + alpha * (
                    #             np.random.rand(num_dimensions) - 0.5)
                    # elite_fireflies[i] = np.clip(elite_fireflies[i], bounds[0], bounds[1])
                    # elite_intensity[i] = sphere(elite_fireflies[i])
            else:
                elite_fireflies[i] = generate_new_solution_by_swapping_features(elite_fireflies[i])
                candidate_eval = objective_function(X,elite_fireflies[i])
                if random.uniform(0, 1) < math.exp(
                    -abs(candidate_eval - objective_function(X,elite_fireflies[j])) / temp):
                    elite_fireflies[i], current_eval = elite_fireflies[i], candidate_eval

                    # 更新最佳解
                    # if candidate_eval < best_eval:
                    #     best_solution, best_eval = elite_fireflies[i], candidate_eval

                    # 降温
                temp *= (1 - cooling_rate)

        # for i in range(num_fireflies - num_elites):
        #     for j in range(num_elites):
        #         # if non_elite_intensity[i] > sphere(centroid):  # i向j移动
        #         if non_elite_intensity[i] > sphere(elite_fireflies[j]):
        #             r = np.linalg.norm(non_elite_fireflies[i] - elite_fireflies[j])
        #             # beta = beta_0 * np.exp(-gamma * r ** 2)
        #             beta = betamin+(beta_0-betamin) * np.exp(-gamma * r ** 2)
        #             non_elite_fireflies[i] += beta * (elite_fireflies[j] - non_elite_fireflies[i]) + alpha * (
        #                         np.random.rand(num_dimensions) - 0.5)
        #             non_elite_fireflies[i] = np.clip(non_elite_fireflies[i], bounds[0], bounds[1])
        #             non_elite_intensity[i] = sphere(non_elite_fireflies[i])

        for i in range(num_fireflies - num_elites):
            # for j in range(num_elites):
                # if non_elite_intensity[i] > sphere(centroid):  # i向j移动
                if non_elite_intensity[i] > objective_function(X,centroid):
                    r = np.linalg.norm(non_elite_fireflies[i] - centroid)
                    beta = betamin+(beta_0-betamin) * np.exp(-gamma * r ** 2)
                    non_elite_fireflies[i] += beta * (centroid - non_elite_fireflies[i]) + alpha * (
                                np.random.rand(num_dimensions) - 0.5)
                    non_elite_fireflies[i] = np.clip(non_elite_fireflies[i], bounds[0], bounds[1])
                    non_elite_intensity[i] = objective_function(X,non_elite_fireflies[i])


        # 更新整个群体
        fireflies = np.vstack((elite_fireflies, non_elite_fireflies))
        light_intensity = np.hstack((elite_intensity, non_elite_intensity))

        # 记录最佳适应度
        my_best_fitness_history.append(light_intensity[0])
        # 输出当前最佳解

        print(f"Iteration {iter}: Best Fitness = {light_intensity[0]}")
    return light_intensity[0]



# 萤火虫算法主循环
results = []
count = 0
max_value = float('-inf')  # 初始化为负无穷大
min_value = float('inf')   # 初始化为正无穷大
for _ in range(20):
    best_value = myFA()
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

