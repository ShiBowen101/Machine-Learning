import numpy as np


# 总体而言，Python是把树这一数据类型转化为类，而把具体的数据结构转化为数组的中信息的先后顺序
# 一个典型例子
# 定义一个TreeNode类来表示树的节点
# 定义一个TreeNode类来表示树的节点
# class TreeNode:
#     def __init__(self, value):
#         """
#         初始化方法，创建一个新的树节点。
#         :param value: 节点存储的数据
#         """
#         self.value = value  # 存储节点值
#         self.children = []  # 存储子节点列表
#
#     def add_child(self, child_node):
#         """
#         添加一个子节点到当前节点的子节点列表中。
#         :param child_node: 要添加的子节点
#         """
#         self.children.append(child_node)
#
#     def remove_child(self, child_node):
#         """
#         从当前节点的子节点列表中移除一个子节点。
#         :param child_node: 要移除的子节点
#         """
#         self.children.remove(child_node)
#
#     def __repr__(self, level=0):
#         """
#         返回一个字符串表示整个子树，用于打印和调试。
#         :param level: 当前节点深度，用于缩进显示
#         :return: 字符串表示的子树
#         """
#         ret = "\t" * level + repr(self.value) + "\n"  # 当前节点的值，根据深度缩进
#         for child in self.children:
#             ret += child.__repr__(level + 1)  # 递归地调用子节点的__repr__方法，增加缩进
#         return ret
#
# # 创建根节点
# root = TreeNode("root")
#
# # 创建子节点
# child1 = TreeNode("child1")
# child2 = TreeNode("child2")
#
# # 将子节点添加到根节点
# root.add_child(child1)
# root.add_child(child2)
#
# # 创建孙节点
# grandchild1 = TreeNode("grandchild1")
# grandchild2 = TreeNode("grandchild2")
#
# # 将孙节点添加到子节点
# child1.add_child(grandchild1)
# child1.add_child(grandchild2)
#
# # 打印树形结构
# print(root)


# 以决策树为例
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


data = np.array([[1, 2, 3], [2, 3, 4], [5, 6, 7]])
target = np.array([1, 2, 3])
target = target.transpose()
root = Tree(data, target)
node1 = Tree(data, target)
node2 = Tree(data, target)
node3 = Tree(data, target)
node4 = Tree(data, target)
node5 = Tree(data, target)
root.tree_add(node1)
root.tree_add(node2)
node1.tree_add(node3)
node2.tree_add(node4)
node3.tree_add(node5)
print(root)
