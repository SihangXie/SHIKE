# -*- coding:utf-8 -*-
"""
@Project: SHIKE
@File：scheduler.py
@Author：Sihang Xie
@Time：2024/2/27 13:30
@Description：切片网络调度算法测试
"""

import pulp
import numpy as np
import copy

# 初始化参数
n = 5  # 流的数量
m = 3  # 转发设备的数量
C = [100, 80, 120]  # 每个转发设备的总带宽容量
alpha = 1.0  # 随时间的经验值前的超参数
replacement = 1  # 惩罚项p(i)的初始值
P = 10  # TODO 重新分配流所产生的惩罚值
time_periods = 4  # 时间周期数
current_capacity = copy.deepcopy(C)  # 设备的实时剩余带宽
device_stream_list = [[], [], []]  # 每个设备上的流

# 模拟每条流在每个时间周期的带宽需求变化
B_time = np.random.randint(20, 50, size=(n, time_periods))

# 初始化统计数据s
bandwidth_utilization_ratios = []  # 带宽利用率
relocation_counts = []  # 重新分配的次数

for t in range(time_periods):
    B = B_time[:, t].tolist()  # 当前时间周期的带宽需求

    # 创建问题实例
    model = pulp.LpProblem("Bandwidth_Allocation", pulp.LpMaximize)

    # 定义决策变量
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(m)), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", ((i, j, k) for i in range(n) for j in range(m) for k in range(m) if j != k), cat=pulp.LpBinary)

    # TODO 计算惩罚因子
    if t == 0:
        P = [0] * n
    else:
        P = []
        f_t = alpha * ((t / n) ** 2)
        replacement = 1 if sum(relocation_counts) == 0 else sum(relocation_counts)
        b_i, r_i = 0, 0
        B_prev = B_time[:, t - 1]
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    if j != k and y[i, j, k].varValue > 0.0:
                        # 统计b_i值的数量
                        for b in B_prev:
                            if 0.95 <= float(B_prev[i]) / float(b) <= 1.05:
                                b_i += 1
                        # 流i目前所在设备上的流
                        for s in device_stream_list[k]:
                            # 被调离时设备上剩余带宽
                            if 0.95 <= float(current_capacity[j]) / float(s) <= 1.05:
                                r_i += 1
            # 计算p(i)
            p_i: float = float(b_i * r_i) / float(replacement ** 2)
            punish = p_i * f_t
            P.append(punish)

    # 目标函数：最大化带宽利用总和，减去重新分配产生的惩罚
    model += pulp.lpSum([x[i, j] * B[i] for i in range(n) for j in range(m)]) - \
             pulp.lpSum([y[i, j, k] * P[i] for i in range(n) for j in range(m) for k in range(m) if j != k])

    # 约束条件
    # 每条流只能被分配给一个转发设备
    for i in range(n):
        model += pulp.lpSum([x[i, j] for j in range(m)]) == 1

    # 设备的带宽容量约束
    for j in range(m):
        model += pulp.lpSum([x[i, j] * B[i] for i in range(n)]) + \
                 pulp.lpSum(y[i, j, k] * B[i] for i in range(n) for j in range(m) for k in range(m) if j != k) - \
                 pulp.lpSum(y[i, l, j] * B[i] for i in range(n) for l in range(m) for j in range(m) if l != j) <= C[j]

    # 流只有在其原设备超载时才被重新分配
    for i in range(n):
        for j in range(m):
            model += pulp.lpSum([y[i, j, k] for k in range(m) if k != j]) <= x[i, j]

    # 求解
    model.solve()

    # 更新设备状态
    for i in range(n):
        for j in range(m):
            if x[i, j].varValue > 0:
                current_capacity[j] -= B[i]  # 设备j扣掉容量
                device_stream_list[j].append(B[i])  # 设备j新增流i
            for k in range(m):
                if j != k and y[i, j, k].varValue > 0:
                    current_capacity[j] += B[i]  # 设备j增加调走的容量
                    device_stream_list[j].remove(B[i])  # 设备j移除流i
                    current_capacity[k] -= B[i]  # 设备k扣除调入的容量
                    device_stream_list[k].append(B[i])  # 设备k新增流i

    # 统计带宽利用率
    total_bandwidth_utilized = sum([pulp.value(x[i, j]) * B[i] for i in range(n) for j in range(m)])
    total_bandwidth_capacity = sum(C)
    bandwidth_utilization_ratio = total_bandwidth_utilized / total_bandwidth_capacity
    bandwidth_utilization_ratios.append(bandwidth_utilization_ratio)

    # 统计调出次数
    relocation_count = sum([pulp.value(y[i, j, k]) for i in range(n) for j in range(m) for k in range(m) if j != k])
    relocation_counts.append(relocation_count)

    print(f"Time Period {t}: Bandwidth Utilization Ratio = {bandwidth_utilization_ratio:.2f}, Relocation Count = {relocation_count}")

# 总结统计信息
average_bandwidth_utilization_ratio = np.mean(bandwidth_utilization_ratios)
average_relocation_count = np.mean(relocation_counts)
print(f"Average Bandwidth Utilization Ratio over time: {average_bandwidth_utilization_ratio:.2f}")
print(f"Average Relocation Count over time: {average_relocation_count}")
