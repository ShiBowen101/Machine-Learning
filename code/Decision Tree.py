import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# information entropy(信息熵) Ent=sum(-P_k*log2(P_k))其中P_k为某类别样本在决策分支的占比
# 本程序以信息熵作为判别依据
set = load_iris()
data = set['data']
target = set['target']
size0 = 0
size1 = 0
size2 = 0
for letter in target:
    if letter == 0:
        size0 = size0 + 1
    if letter == 1:
        size1 = size1 + 1
    if letter == 2:
        size2 = size2 + 1
print(size0, size1, size2)
# plt.figure()
# plt.plot(data[0:size0, 0], data[0:size0, 1], 'o')
# plt.plot(data[size0:size1 + size0, 0], data[size0:size1 + size0, 1], 'o')
# plt.plot(data[size1 + size0:size2 + size1 + size0, 0], data[size1 + size0:size2 + size1 + size0, 1], 'o')
# plt.show()
# 来获取数据的大致分布信息
size = np.size(target)
# size=log2(n)一般为了防止过拟合设定的最大深度
max_size = int(math.log2(size))
class_size = np.size(set['target_names'])  # 类别数目


# print(np.argsort(data[:,0]))#此函数的作用是返回排序后的索引。
# 开始训练
# 适用于连续变量


def split(test_data, test_target, target_name_size):
    ent = 0
    min_ent = 99
    index_stack = np.argsort(test_data)
    f_stack = np.sort(test_data)
    data_size = np.size(test_data)
    storage_boundary = -1

    for i in range(0, data_size, 1):
        if i == data_size-3:
            break
        boundary = (f_stack[i] + f_stack[i + 1]) / 2
        count1 = np.zeros(target_name_size)
        count2 = np.zeros(target_name_size)
        for one in index_stack[i + 1:]:
            for j in range(0, target_name_size, 1):
                if j == target[one]:
                    count1[j] = count1[j] + 1
        for one in index_stack[:i + 1]:
            for j in range(0, target_name_size, 1):
                if j == target[one]:
                    count2[j] = count2[j] + 1
        p_2 = i / data_size
        p_1 = 1 - p_2
        sum1 = 0
        sum2 = 0
        for one in range(0, target_name_size, 1):
            if count1[one] == 0:
                continue
            sum1 = count1[one] / (data_size - i) * math.log2(count1[one] / (data_size - i)) + sum1
        for one in range(0, target_name_size, 1):
            if count2[one] == 0:
                continue
            sum2 = count2[one] / i * math.log2(count2[one] / i) + sum2

        ent = -(p_1 * sum1 + p_2 * sum2)
        if min_ent > ent:
            min_ent = ent
            storage_boundary = i
    target_data1 = np.zeros(storage_boundary)
    target_data2 = np.zeros(data_size - storage_boundary)
    target_1 = np.zeros(storage_boundary)
    target_2 = np.zeros(data_size - storage_boundary)
    for i in range(0, storage_boundary, 1):
        target_data1[i] = test_data[index_stack[i]]
        target_1[i] = test_target[index_stack[i]]
    for i in range(0, data_size - storage_boundary, 1):
        target_data2[i] = test_data[index_stack[i + storage_boundary]]
        target_2[i] = test_target[index_stack[i + storage_boundary]]
    return target_data1, target_1, target_data2, target_2


(target1d, target1t, target2d, target2t) = split(data[:, 0], target, class_size)
#搁置
