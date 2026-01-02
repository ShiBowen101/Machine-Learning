import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# 本程序示例为连续型变量

class Tree:
    def __init__(self, value, target):
        self.value = value
        self.target = target
        self.nodes_link = []  # 用于存储子节点

    def tree_add(self, child):
        self.nodes_link.append(child)

    def tree_delete(self, child):
        self.nodes_link.remove(child)

    # 这一部分是用于打印的
    def __repr__(self, level=0):
        ret = "\t" * level + f"Tree( {self.target})\n"
        for child in self.nodes_link:
            ret += child.__repr__(level + 1)
        return ret


# 判断子树中的类别是否唯一
def judge_one_or_more(data):
    stack = data[0]
    for one in data:
        if one != stack:
            return 1
    return 0


def get_the_class_of_target(target):
    class_list = []
    for one in target:
        if np.size(class_list) == 0:
            class_list.append(one)
        else:
            sign = 0
            for two in class_list:
                if two == one:
                    sign = 1
                    break
            if sign == 0:
                class_list.append(one)
    return class_list


def choose_the_best_feature(data, target):
    size_feature = np.size(data[0, :])
    # 按照特征进行分组
    class_list = get_the_class_of_target(target)
    class_size = np.size(class_list)
    ent = 0
    ent_min = 10
    compare_ent = np.zeros(size_feature)
    compare_boundary = np.zeros(size_feature)
    ent_min = 10
    for i in range(0, size_feature, 1):
        stack = data[:, i]
        boundary_choose = np.sort(stack)
        index_boundary = np.argsort(stack)
        for j in range(0, np.size(boundary_choose) - 1, 1):
            a = np.size(boundary_choose) - 1
            ent = 0
            count_part1 = np.zeros(class_size)
            count_part2 = np.zeros(class_size)
            boundary = (boundary_choose[j] + boundary_choose[j + 1]) / 2
            part1_index = range(0, j + 1, 1)
            part2_index = range(j + 1, np.size(boundary_choose), 1)
            for one in part1_index:
                for two in range(0, class_size, 1):
                    if target[index_boundary[one]] == class_list[two]:
                        count_part1[two] += 1  # 也许待验证
            for one in part2_index:
                for two in range(0, class_size, 1):
                    if target[index_boundary[one]] == class_list[two]:
                        count_part2[two] += 1  # 也许待验证
            # 计算单元
            for w in range(0, class_size, 1):
                if count_part1[w] == 0:
                    continue
                ent = (j + 1) / np.size(boundary_choose) * (
                        count_part1[w] / (j + 1) * math.log2(count_part1[w] / (j + 1))) \
                      + ent  # 换行符
                # 加权
            for v in range(0, class_size, 1):
                if count_part2[v] == 0:
                    continue
                ent = ((np.size(boundary_choose) - j - 1) / np.size(boundary_choose)) \
                      * (count_part2[v] / (np.size(boundary_choose) - j - 1) *
                         math.log2(count_part2[v] / (np.size(boundary_choose) - j - 1))) + ent
            ent = -ent
            if ent_min > ent:
                ent_min = ent
                compare_boundary[i] = boundary
                compare_ent[i] = ent_min
    # 未产生最值的迭代回合对应数值会显示为0，因此对数据0进行处理
    for letter1 in range(0, np.size(compare_ent), 1):
        if compare_ent[letter1] == 0:
            b = compare_ent[letter1]
            compare_ent[letter1] = 100
    index_compare = np.argsort(compare_ent)
    compare_ent = np.sort(compare_ent)
    the_index_of_feature = index_compare[0]
    the_boundary_of_feature = compare_boundary[the_index_of_feature]
    return the_index_of_feature, the_boundary_of_feature
    # 完毕


depth = 0


def create_tree(data, target, node):
    sign1 = judge_one_or_more(target)
    go_on = 1
    if np.size(target) < 1:  # 确保存在数据
        return 0
    if sign1 == 0:  # 此时表示纯
        return 0
    global depth  # 通过声明全局变量来替换静态变量
    depth += 1
    if depth > 20:
        return 0
    (the_index_of_feature, the_boundary_of_feature) = choose_the_best_feature(data, target)
    feature_stack = data[:, the_index_of_feature]
    index_feature = np.argsort(feature_stack)
    feature_stack = np.sort(feature_stack)
    index_boundary = 0
    for one in feature_stack:
        if one > the_boundary_of_feature:
            break
        index_boundary = index_boundary + 1
    child_value1 = np.zeros((index_boundary + 1, np.size(data[0, :])))
    child_value2 = np.zeros((np.size(target) - index_boundary - 1, np.size(data[0, :])))
    child_target1 = np.zeros(index_boundary + 1)
    child_target2 = np.zeros(np.size(target) - index_boundary - 1)
    count = 0
    w = 0
    v = 0
    for two in index_feature:
        if count > index_boundary:
            child_value2[w, :] = data[two, :]
            child_target2[w] = target[two]
            w = w + 1
        else:
            child_value1[v, :] = data[two, :]
            child_target1[v] = target[two]
            v = v + 1
        count = count + 1
    # 分裂成功，待调试
    child1 = Tree(child_value1, child_target1)
    child2 = Tree(child_value2, child_target2)
    node.tree_add(child1)
    node.tree_add(child2)
    create_tree(child_value1, child_target1, child1)
    create_tree(child_value2, child_target2, child2)
    # 递归部分


# information entropy(信息熵) Ent=sum(-P_k*log2(P_k))其中P_k为某类别样本在决策分支的占比
sets = datasets.load_iris()
data = sets['data']
target = sets['target']
root = Tree(data, target)
create_tree(data, target, root)  # 其实create_tree函数的data，target参数是多余的
# 慢慢调试吧，目前不具备决策功能
print('ok')
# 问题：judege溢出
