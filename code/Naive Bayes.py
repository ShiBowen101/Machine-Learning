import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import sklearn as sk

# 数据处理模块
'''
matrix = np.array(np.arange(1, 17, 1))
matrix = matrix.reshape(4, 4)
print(matrix)
dates = (pd.date_range("20231004", periods=6))
matrix1 = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['a', 'b', 'c', 'd'])
print(matrix1.describe())  # std 标准差
value = matrix1.describe()
print(value.loc['std'])
sum = value.loc['count'] * value.loc['mean']
print(sum)  # 总数
'''
# 数据提取模块
# 青绿 0 乌黑 1 浅白 2
# 蜷缩 0 稍蜷 1 硬挺 2
# 浊响 0 沉闷 1 清脆 2
# 清晰 0 稍糊 1 模糊 2
# 凹陷 0 稍凹 1 平坦 2
# 硬滑 0 软粘 1
# 是  0 好  1
class_color_00 = 0
class_color_01 = 0
class_color_10 = 0
class_color_11 = 0
class_color_20 = 0
class_color_21 = 0
class_root_00 = 0
class_root_01 = 0
class_root_10 = 0
class_root_11 = 0
class_root_20 = 0
class_root_21 = 0
class_hoot_00 = 0  # hoot敲声
class_hoot_01 = 0
class_hoot_10 = 0
class_hoot_11 = 0
class_hoot_20 = 0
class_hoot_21 = 0
class_texture_00 = 0  # texture纹理
class_texture_01 = 0
class_texture_10 = 0
class_texture_11 = 0
class_texture_20 = 0
class_texture_21 = 0
class_qi_00 = 0
class_qi_01 = 0
class_qi_10 = 0
class_qi_11 = 0
class_qi_20 = 0
class_qi_21 = 0
class_touch_00 = 0
class_touch_01 = 0
class_touch_10 = 0
class_touch_11 = 0
data = pd.read_excel('data3.0.xlsx')
print(data)
i = 0
for letter in data.loc[:, "色泽"]:
    if letter == "青绿":
        data.iloc[i, 1] = 0
    elif letter == "乌黑":
        data.iloc[i, 1] = 1
    elif letter == "浅白":
        data.iloc[i, 1] = 2
    i = i + 1
i = 0
for letter in data.loc[:, "根蒂"]:
    if letter == "蜷缩":
        data.iloc[i, 2] = 0
    elif letter == "稍蜷":
        data.iloc[i, 2] = 1
    elif letter == "硬挺":
        data.iloc[i, 2] = 2
    i = i + 1
i = 0
for letter in data.loc[:, "敲声"]:
    if letter == "浊响":
        data.iloc[i, 3] = 0
    elif letter == "沉闷":
        data.iloc[i, 3] = 1
    elif letter == "清脆":
        data.iloc[i, 3] = 2
    i = i + 1
i = 0
for letter in data.loc[:, "纹理"]:
    if letter == "清晰":
        data.iloc[i, 4] = 0
    elif letter == "稍糊":
        data.iloc[i, 4] = 1
    elif letter == "模糊":
        data.iloc[i, 4] = 2
    i = i + 1
i = 0
for letter in data.loc[:, "脐部"]:
    if letter == "凹陷":
        data.iloc[i, 5] = 0
    elif letter == "稍凹":
        data.iloc[i, 5] = 1
    elif letter == "平坦":
        data.iloc[i, 5] = 2
    i = i + 1
i = 0
for letter in data.loc[:, "触感"]:
    if letter == "硬滑":
        data.iloc[i, 6] = 0
    elif letter == "软粘":
        data.iloc[i, 6] = 1
    i = i + 1
i = 0
for letter in data.loc[:, "好瓜"]:
    if letter == "是":
        data.iloc[i, 9] = 0
    elif letter == "否":
        data.iloc[i, 9] = 1
    i = i + 1
print(data)
# 数据预处理模块
count0 = 0
count1 = 0
count2 = 0
count_0 = 0
count_1 = 0
count_2 = 0
temp = data['好瓜'].value_counts()
P_0 = temp[0] / (temp[0] + temp[1])
P_1 = 1 - P_0
print(temp)
for one in data['编号']:
    if data.iloc[one - 1, 9] == 0:
        if data.iloc[one - 1, 1] == 0:
            count0 = count0 + 1
        if data.iloc[one - 1, 1] == 1:
            count1 = count1 + 1
        if data.iloc[one - 1, 1] == 2:
            count2 = count2 + 1
    else:
        if data.iloc[one - 1, 1] == 0:
            count_0 = count_0 + 1
        if data.iloc[one - 1, 1] == 1:
            count_1 = count_1 + 1
        if data.iloc[one - 1, 1] == 2:
            count_2 = count_2 + 1
print("11111111111111111111111111111")
print(count0, count1, count2)
print(data)
class_color_00 = (count0 + 1) / (temp[0] + 3)  # Laplace Smooth
class_color_01 = (count_0 + 1) / (temp[1] + 3)
class_color_10 = (count1 + 1) / (temp[0] + 3)
class_color_11 = (count_1 + 1) / (temp[1] + 3)
class_color_20 = (count2 + 1) / (temp[0] + 3)
class_color_21 = (count_2 + 1) / (temp[1] + 3)
count0 = 0
count1 = 0
count2 = 0
count_0 = 0
count_1 = 0
count_2 = 0
for one in data['编号']:
    if data.iloc[one - 1, 9] == 0:
        if data.iloc[one - 1, 2] == 0:
            count0 = count0 + 1
        if data.iloc[one - 1, 2] == 1:
            count1 = count1 + 1
        if data.iloc[one - 1, 2] == 2:
            count2 = count2 + 1
    else:
        if data.iloc[one - 1, 2] == 0:
            count_0 = count_0 + 1
        if data.iloc[one - 1, 2] == 1:
            count_1 = count_1 + 1
        if data.iloc[one - 1, 2] == 2:
            count_2 = count_2 + 1
class_root_00 = (count0 + 1) / (temp[0] + 3)
class_root_01 = (count_0 + 1) / (temp[1] + 3)
class_root_10 = (count1 + 1) / (temp[0] + 3)
class_root_11 = (count_1 + 1) / (temp[1] + 3)
class_root_20 = (count2 + 1) / (temp[0] + 3)
class_root_21 = (count_2 + 1) / (temp[1] + 3)
count0 = 0
count1 = 0
count2 = 0
count_0 = 0
count_1 = 0
count_2 = 0
for one in data['编号']:
    if data.iloc[one - 1, 9] == 0:
        if data.iloc[one - 1, 3] == 0:
            count0 = count0 + 1
        if data.iloc[one - 1, 3] == 1:
            count1 = count1 + 1
        if data.iloc[one - 1, 3] == 2:
            count2 = count2 + 1
    else:
        if data.iloc[one - 1, 3] == 0:
            count_0 = count_0 + 1
        if data.iloc[one - 1, 3] == 1:
            count_1 = count_1 + 1
        if data.iloc[one - 1, 3] == 2:
            count_2 = count_2 + 1
class_hoot_00 = (count0 + 1) / (temp[0] + 3)
class_hoot_01 = (count_0 + 1) / (temp[1] + 3)
class_hoot_10 = (count1 + 1) / (temp[0] + 3)
class_hoot_11 = (count_1 + 1) / (temp[1] + 3)
class_hoot_20 = (count2 + 1) / (temp[0] + 3)
class_hoot_21 = (count_2 + 1) / (temp[1] + 3)
count0 = 0
count1 = 0
count2 = 0
count_0 = 0
count_1 = 0
count_2 = 0
for one in data['编号']:
    if data.iloc[one - 1, 9] == 0:
        if data.iloc[one - 1, 4] == 0:
            count0 = count0 + 1
        if data.iloc[one - 1, 4] == 1:
            count1 = count1 + 1
        if data.iloc[one - 1, 4] == 2:
            count2 = count2 + 1
    else:
        if data.iloc[one - 1, 4] == 0:
            count_0 = count_0 + 1
        if data.iloc[one - 1, 4] == 1:
            count_1 = count_1 + 1
        if data.iloc[one - 1, 4] == 2:
            count_2 = count_2 + 1
class_texture_00 = (count0 + 1) / (temp[0] + 3)
class_texture_01 = (count_0 + 1) / (temp[1] + 3)
class_texture_10 = (count1 + 1) / (temp[0] + 3)
class_texture_11 = (count_1 + 1) / (temp[1] + 3)
class_texture_20 = (count2 + 1) / (temp[0] + 3)
class_texture_21 = (count_2 + 1) / (temp[1] + 3)
count0 = 0
count1 = 0
count2 = 0
count_0 = 0
count_1 = 0
count_2 = 0
for one in data['编号']:
    if data.iloc[one - 1, 9] == 0:
        if data.iloc[one - 1, 5] == 0:
            count0 = count0 + 1
        if data.iloc[one - 1, 5] == 1:
            count1 = count1 + 1
        if data.iloc[one - 1, 5] == 2:
            count2 = count2 + 1
    else:
        if data.iloc[one - 1, 5] == 0:
            count_0 = count_0 + 1
        if data.iloc[one - 1, 5] == 1:
            count_1 = count_1 + 1
        if data.iloc[one - 1, 5] == 2:
            count_2 = count_2 + 1
class_qi_00 = (count0 + 1) / (temp[0] + 3)
class_qi_01 = (count_0 + 1) / (temp[1] + 3)
class_qi_10 = (count1 + 1) / (temp[0] + 3)
class_qi_11 = (count_1 + 1) / (temp[1] + 3)
class_qi_20 = (count2 + 1) / (temp[0] + 3)
class_qi_21 = (count_2 + 1) / (temp[1] + 3)
count0 = 0
count1 = 0
count2 = 0
count_0 = 0
count_1 = 0
count_2 = 0
for one in data['编号']:
    if data.iloc[one - 1, 9] == 0:
        if data.iloc[one - 1, 6] == 0:
            count0 = count0 + 1
        if data.iloc[one - 1, 6] == 1:
            count1 = count1 + 1
    else:
        if data.iloc[one - 1, 6] == 0:
            count_0 = count_0 + 1
        if data.iloc[one - 1, 6] == 1:
            count_1 = count_1 + 1
class_touch_00 = (count0 + 1) / (temp[0] + 2)
class_touch_01 = (count_0 + 1) / (temp[1] + 2)
class_touch_10 = (count1 + 1) / (temp[0] + 2)
class_touch_11 = (count_1 + 1) / (temp[1] + 2)
# 连续变量部分
class_0 = data.loc[0:8, :]
class_1 = data.loc[9:16, :]
calculation_0 = class_0.describe()
calculation_1 = class_1.describe()
density_mean_0 = calculation_0.loc['mean', '密度']  # density密度
sugar_mean_0 = calculation_0.loc['mean', '含糖量']
density_std_0 = calculation_0.loc['std', '密度']
sugar_std_0 = calculation_0.loc['std', '含糖量']
density_mean_1 = calculation_1.loc['mean', '密度']
sugar_mean_1 = calculation_1.loc['mean', '含糖量']
density_std_1 = calculation_1.loc['std', '密度']
sugar_std_1 = calculation_1.loc['std', '含糖量']
# 交互界面
while 1:
    print("请输入特征")
    print("1.色泽(青绿 0 乌黑 1 浅白 2):")
    color = input()
    print("2.根蒂(蜷缩 0 稍蜷 1 硬挺 2):")
    root = input()
    print("3.敲声(浊响 0 沉闷 1 清脆 2):")
    hoot = input()
    print("4.纹理(清晰 0 稍糊 1 模糊 2):")
    texture = input()
    print("5.脐部(凹陷 0 稍凹 1 平坦 2):")
    qi = input()
    print("6.触感(硬滑 0 软粘 1):")
    touch = input()
    print("7.密度：")
    density = input()
    print("8.含糖量：")
    sugar = input()
    P_color = 0
    P_color_1 = 0
    P_root = 0
    P_root_1 = 0
    P_hoot = 0
    P_hoot_1 = 0
    P_texture = 0
    P_texture_1 = 0
    P_qi = 0
    P_qi_1 = 0
    P_touch = 0
    P_touch_1 = 0
    # 离散变量部分
    if color == '0':
        P_color = class_color_00
        P_color_1 = class_color_01
    elif color == '1':
        P_color = class_color_10
        P_color_1 = class_color_11
    else:
        P_color = class_color_20
        P_color_1 = class_color_21
    if root == '0':
        P_root = class_root_00
        P_root_1 = class_root_01
    elif root == '1':
        P_root = class_root_10
        P_root_1 = class_root_11
    else:
        P_root = class_root_20
        P_root_1 = class_root_21
    if hoot == '0':
        P_hoot = class_hoot_00
        P_hoot_1 = class_hoot_01
    elif hoot == '1':
        P_hoot = class_hoot_10
        P_hoot_1 = class_hoot_11
    else:
        P_hoot = class_hoot_20
        P_hoot_1 = class_hoot_21
    if texture == '0':
        P_texture = class_texture_00
        P_texture_1 = class_texture_01
    elif texture == '1':
        P_texture = class_texture_10
        P_texture_1 = class_texture_11
    else:
        P_texture = class_texture_20
        P_texture_1 = class_texture_21
    if qi == '0':
        P_qi = class_qi_00
        P_qi_1 = class_qi_01
    elif qi == '1':
        P_qi = class_qi_10
        P_qi_1 = class_qi_11
    else:
        P_qi = class_qi_20
        P_qi_1 = class_qi_21
    if touch == '0':
        P_touch = class_touch_00
        P_touch_1 = class_touch_01
    else:
        P_touch = class_touch_10
        P_touch_1 = class_touch_11
    # 连续变量部分
    P_density_0 = (1 / (math.pow(2 * math.pi, 0.5) * density_std_0) *
                   math.exp(-math.pow(float(density) - density_mean_0, 2) / (2 * density_std_0 * density_std_0)))
    P_density_1 = (1 / (math.pow(2 * math.pi, 0.5) * density_std_1) *
                   math.exp(-math.pow(float(density) - density_mean_1, 2) / (2 * density_std_1 * density_std_1)))
    P_sugar_0 = (1 / (math.pow(2 * math.pi, 0.5) * sugar_std_0) *
                 math.exp(-math.pow(float(sugar) - sugar_mean_0, 2) / (2 * sugar_std_0 * sugar_std_0)))
    P_sugar_1 = (1 / (math.pow(2 * math.pi, 0.5) * sugar_std_1) *
                 math.exp(-math.pow(float(sugar) - sugar_mean_1, 2) / (2 * sugar_std_1 * sugar_std_1)))
    '''
    P_yes = (math.log(P_color) + math.log(P_root) + math.log(P_hoot) + math.log(P_texture) + math.log(P_touch) +
              math.log(P_qi) + math.log(P_0) + math.log(P_sugar_0) + math.log(P_density_0))  # 连乘很容易造成下溢
    P_no = (math.log(P_color_1) + math.log(P_root_1) + math.log(P_hoot_1) + math.log(P_texture_1) +
             math.log(P_touch_1) + math.log(P_qi_1) + math.log(P_1te) + math.log(P_sugar_1) + math.log(P_density_1))'''
    P_yes = P_color * P_root * P_hoot * P_texture * P_touch * P_qi * P_0 * P_sugar_0 * P_density_0
    P_no = P_color_1 * P_root_1 * P_hoot_1 * P_texture_1 * P_touch_1 * P_qi_1 * P_1 * P_sugar_1 * P_density_1
    P0 = P_yes / (P_yes + P_no)
    P1 = 1 - P0
    print(P0, P1)
    if P_yes >= P_no:
        print("这是个好瓜")
    else:
        print("这瓜不行")
# 输入类型默认为字符型
# 收工
