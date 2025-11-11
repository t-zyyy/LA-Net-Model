import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#使系统智能看到编号为1的GPU
import matplotlib.pyplot as plt
# 设置全局字体为 Times New Roman
# plt.rcParams['font.family'] = 'Nimbus Roman'
# plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'  # 或者使用其他你选择的字体名
plt.rcParams['font.family'] = 'SimHei'
# import pandas as pd
#
# # 读取Excel文件
# df = pd.read_excel('/home/zsq/train/train.xlsx')
#
# # 打印列名
# print(df.columns)
#
# # 统计B列中数字出现的次数
# # count = df['B'].value_counts()
# # 使用列索引或列名获取 B 列
# count = df['B'].value_counts().sort_index()
#
# # 输出结果
# print(count)
#
# # 计算出现次数的总和
# total_count = count.sum()
#
# # 输出总和
# print(f"总共出现的次数：{total_count}")











# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置中文字体

# rcParams.update({
#      'figure.dpi': 100,  # 降低 DPI 以避免显示问题
# })
#
# # 指定存放 Excel 文件的文件夹路径
# folder_path = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图"  # 请替换为实际路径
#
# # 获取文件夹下所有 Excel 文件
# excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
#
# # 初始化空列表存储数据
# data = []
#
# # 遍历所有 Excel 文件
# for file in excel_files:
#     file_path = os.path.join(folder_path, file)
#     group_name = os.path.splitext(file)[0]  # 以文件名作为组名
#
#     try:
#         # 读取 Excel 文件（假设前两列是类别数据）
#         df = pd.read_excel(file_path, usecols=[0, 1])  # 只读取前两列，避免格式问题
#
#         if df.shape[1] < 2:
#             print(f"⚠️ 文件 {file} 格式不符合要求，跳过")
#             continue
#
#         # 获取两列的列名（如 "Deceased" 和 "Surviving"）
#         left_category, right_category = df.columns[:2]
#
#         # 转换数据格式（长格式）
#         for value in df[left_category].dropna():
#             data.append((group_name, str(left_category), value))  # 确保 Status 为字符串
#         for value in df[right_category].dropna():
#             data.append((group_name, str(right_category), value))
#
#     except Exception as e:
#         print(f"❌ 处理文件 {file} 时出错：{e}")
#         continue
#
# # 转换为 DataFrame
# df_all = pd.DataFrame(data, columns=["Group", "Status", "Body Age Gap"])
#
# # 确保 "Status" 列是字符串
# df_all["Status"] = df_all["Status"].astype(str)
#
# # 创建自定义x轴位置
# custom_x_positions = [-0.5, 0, 1]  # 自定义每组小提琴图的x轴位置
# group_to_position = {group: pos for group, pos in zip(df_all["Group"].unique(), custom_x_positions)}
# df_all['Group_pos'] = df_all['Group'].map(group_to_position)
#
# # 画小提琴图
# plt.figure(figsize=(12, 6))
#
# # 定义左侧和右侧的颜色
# left_colors = ['#87CEEB', '#87CEEB', '#87CEEB']  # 示例中的浅蓝到深蓝渐变色
# right_colors = ['#FFC0CB', '#FFC0CB', '#FFC0CB']  # 示例中的浅红到深红渐变色
#
# # 确保左右颜色数量与组数匹配
# groups = df_all["Group"].unique()
# num_groups = len(groups)
# if num_groups > len(left_colors) or num_groups > len(right_colors):
#     raise ValueError("需要定义更多颜色以匹配组的数量")
#
# # 创建小提琴图，使用 split=True 进行左右分割，并基于自定义位置绘图
# ax = sns.violinplot(x="Group_pos", y="Body Age Gap", hue="Status", data=df_all,
#                     split=True, inner="quartile", width=0.5)
#
# # 清除默认颜色设置，以便我们可以手动设置颜色
# for collection in ax.collections:
#     collection.set_facecolor('None')
#
# # 手动为每个小提琴设置颜色
# for i, violin in enumerate(ax.collections):
#     # 判断是该组的第一个还是第二个小提琴(左侧或右侧)
#     if i % 2 == 0:  # 偶数索引是左侧的小提琴
#         color = left_colors[i // 2]
#     else:           # 奇数索引是右侧的小提琴
#         color = right_colors[i // 2]
#
#     # 设置小提琴的颜色
#     violin.set_facecolor(color)
#
# # 计算均值并添加标记
# group_means = df_all.groupby(["Group", "Status"])["Body Age Gap"].mean().reset_index()
#
# # 添加均值标记
# for group in groups:
#     for status in df_all["Status"].unique():
#         mean_val = group_means[(group_means["Group"] == group) & (group_means["Status"] == status)]["Body Age Gap"].values
#         if len(mean_val) > 0:
#             plt.scatter(group_to_position[group], mean_val[0], color='black', marker="none", s=100, zorder=3)
#
# # 添加参考线
# plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
#
# # 更新x轴刻度标签
# plt.xticks(ticks=custom_x_positions, labels=groups, rotation=30)
#
# # 设置标题和样式
# plt.title("Body Age Gap by Group and Status")
# plt.ylim(-25,35)
# plt.legend(title="Status", bbox_to_anchor=(1.05, 1), loc="upper left")  # 调整图例位置
# plt.tight_layout()  # 自动调整布局
#
# plt.show()











# #####COPD vs Health 男 女 全部            小提琴图1          第一-散点
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体



# 设置全局字体大小（基准值，可被局部覆盖）
rcParams.update({
    'figure.dpi': 100,  # 降低 DPI 以避免显示问题
    'font.size': 20,           # 默认字体大小（影响大多数文本）
    'axes.titlesize': 20,      # 标题字体大小
    'axes.labelsize': 20,      # 坐标轴标签大小
    'xtick.labelsize': 20,      # X轴刻度标签大小
    'ytick.labelsize': 20,      # Y轴刻度标签大小
    'legend.fontsize': 20,     # 图例字体大小
})









# 指定存放 Excel 文件的文件夹路径
folder_path = "/disk2/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/COPD-Health"  # 请替换为实际路径

# 获取文件夹下所有 Excel 文件
excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

# 自定义顺序：根据文件名指定顺序
custom_order = ["All", "Male", "Female", "35-47", "48-59", "60-74", "G1", "G2", "G3+G4"]  # 替换为实际文件名（不带扩展名）
excel_files = sorted(excel_files, key=lambda x: custom_order.index(os.path.splitext(x)[0]))

# 初始化空列表存储数据
data = []

# 遍历所有 Excel 文件
for file in excel_files:
    file_path = os.path.join(folder_path, file)
    group_name = os.path.splitext(file)[0]  # 以文件名作为组名

    try:
        # 读取 Excel 文件（假设前两列是类别数据）
        df = pd.read_excel(file_path, usecols=[0, 1])  # 只读取前两列，避免格式问题

        if df.shape[1] < 2:
            print(f"⚠️ 文件 {file} 格式不符合要求，跳过")
            continue

        # 获取两列的列名（如 "COPD" 和 "Health"）
        left_category, right_category = df.columns[:2]

        # 转换数据格式（长格式）
        for value in df[left_category].dropna():
            data.append((group_name, "COPD", value))  # 这里用统一的字符串
        for value in df[right_category].dropna():
            data.append((group_name, "Health", value))

    except Exception as e:
        print(f"❌ 处理文件 {file} 时出错：{e}")
        continue

# 转换为 DataFrame
df_all = pd.DataFrame(data, columns=["Group", "Status", "Lung age gap"])

# 确保 "Status" 列是字符串
df_all["Status"] = df_all["Status"].astype(str)

# 按照自定义顺序排序数据
df_all["Group"] = pd.Categorical(df_all["Group"], categories=custom_order, ordered=True)
df_all = df_all.sort_values("Group")

# 生成 x 轴位置
groups = df_all["Group"].unique()
base_x_positions = list(range(len(groups)))
x_offset = 0.15  # 控制左右小提琴的间距

# 生成映射
group_to_position = {group: pos for group, pos in zip(groups, base_x_positions)}

# 计算新的 x 轴位置，使左右分开
df_all["x_pos"] = df_all.apply(
    lambda row: group_to_position[row["Group"]] - x_offset if row["Status"] == "COPD"
    else group_to_position[row["Group"]] + x_offset, axis=1)

# 画图
plt.figure(figsize=(16, 8))

#######  ,  ,
# 颜色定义
# palette = {"COPD": "#E74C3C", "Health": "#5DADE2"}  # 确保颜色映射正确
palette = {"COPD": "#87CEEB", "Health": "#FFC0CB"}  # 确保颜色映射正确

######原来的红蓝   "#87CEEB"       "#FFC0CB"
# 使用 hue 直接区分类别，保证绘制正确
sns.violinplot(x="Group", y="Lung age gap", hue="Status", data=df_all,
               split=True, inner="quartile", palette=palette, linewidth=1, scale="width", width=0.6,cut=1.7)  # 设置 scale="width"

# plt.ylim(-20,25)

plt.ylim(-23,27.5)
plt.yticks(ticks=[-23,  -17.5, -11.4, -5.7,     0,  5.5,   11,  16.5,  22, 27.5],
           labels=["-20","-15","-10", "-5",   "0", "5", "10", "15", "20", "25"])
# 显示范围（真实刻度）



# 计算均值并添加标记
group_means = df_all.groupby(["Group", "Status"])["Lung age gap"].mean().reset_index()
# for _, row in group_means.iterrows():
#     x_pos = group_to_position[row["Group"]] - x_offset if row["Status"] == "COPD" else group_to_position[row["Group"]] + x_offset
#     plt.scatter(x_pos, row["Lung age gap"], color="black", marker="*", s=100, zorder=3)
#
#     # 在每个小提琴图的两半上方添加均值文本
#     plt.text(x_pos, row["Lung age gap"] + 0.5, f"{row['Lung age gap']:.2f}",  # 调整文本位置和格式
#              ha="center", va="bottom", fontsize=18, color="black")
# 定义每个组的文本垂直和水平偏移量
text_offsets = {
    # 格式: {"Group": {"Status": (x_offset, y_offset)}}
    "All": {"COPD": (-0.1, 0.8), "Health": (0.1, 0.8)},
    "Male": {"COPD": (-0.1, 0.8), "Health": (0.1, 0.8)},
    "Female": {"COPD": (-0.1, 0.8), "Health": (0.1, 0.8)},
    "35-47": {"COPD": (-0.1, 0.7), "Health": (0.1, 0.7)},
    "48-59": {"COPD": (-0.1, 0.8), "Health": (0.1, 0.8)},
    "60-74": {"COPD": (-0.1, 0.9), "Health": (0.1, 0.9)},
    "G1": {"COPD": (-0.1, 0.7), "Health": (0.1, 0.7)},
    "G2": {"COPD": (-0.1, 0.8), "Health": (0.1, 0.8)},
    "G3+G4": {"COPD": (-0.1, 0.9), "Health": (0.1, 0.9)}
}

for _, row in group_means.iterrows():
    base_x = group_to_position[row["Group"]]
    x_pos = base_x - x_offset if row["Status"] == "COPD" else base_x + x_offset

    # 获取当前组的偏移量设置
    offsets = text_offsets.get(row["Group"], {"COPD": (0, 0), "Health": (0, 0)})
    x_offset_text, y_offset_text = offsets[row["Status"]]

    plt.scatter(x_pos, row["Lung age gap"], color="black", marker="*", s=100, zorder=3)

    # 应用自定义偏移
    plt.text(
        x_pos + x_offset_text,  # 水平偏移
        row["Lung age gap"] + y_offset_text,  # 垂直偏移
        f"{row['Lung age gap']:.2f}",
        ha="center",
        va="bottom",
        fontsize=16,
        color="black",
        bbox=dict(facecolor='none', alpha=0.7, pad=0.1, edgecolor='none')
    )


# 添加参考线
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)

# 更新 x 轴刻度
plt.xticks(ticks=base_x_positions, labels=groups, rotation=0)

# 隐藏 x 轴标签
plt.xlabel("")  # 隐藏 "Group" 单词

# 隐藏图例标题
plt.legend(title="")  # 隐藏 "Status" 单词

# 设置标题
plt.title("")

# 定义每个分组的星号数量
star_counts = {
    "All": 3,    # All 分组显示 1 个星号
    "Male": 3,   # Male 分组显示 2 个星号
    "Female": 3, # Female 分组显示 3 个星号
    "35-47": 1,   # 0-47 分组显示 1 个星号
    "48-59": 3,  # 48-59 分组显示 2 个星号
    "60-74": 3,  # 60-74 分组显示 3 个星号
    "G1": 3,     # G1 分组显示 1 个星号
    "G2": 3,     # G2 分组显示 2 个星号
    "G3+G4": 3   # G3+G4 分组显示 3 个星号
}

# 定义每个星号的位置 (x, y)
star_positions = {
    "All": (0.23, -25.2),    # All 分组的星号位置
    "Male": (1.33, -25.2),   # Male 分组的星号位置
    "Female": (2.45, -25.2), # Female 分组的星号位置
    "35-47": (3.32, -25.2),   # 0-47 分组的星号位置
    "48-59": (4.40, -25.2),  # 48-59 分组的星号位置
    "60-74": (5.39, -25.2),   # 60-74 分组的星号位置
    "G1": (6.25, -25.2),     # G1 分组的星号位置
    "G2": (7.25, -25.2),     # G2 分组的星号位置
    "G3+G4": (8.48, -25.2)   # G3+G4 分组的星号位置
}

# 在每个分组标签的右上角添加星号
for group, pos in group_to_position.items():
    if group in star_counts and group in star_positions:
        stars = "*" * star_counts[group]  # 根据字典生成星号
        x, y = star_positions[group]  # 获取自定义的 x 和 y 坐标
        plt.text(x, y, stars,  # 在指定位置绘制星号
                 ha="center", va="bottom", fontsize=18, color="black")



plt.tight_layout()
# output_path = 'C:\\Users\\hp\\Desktop\\临时\\1.png'
# plt.savefig(output_path, dpi=600)  # 保存时仍用高DPI
# print(f"图像已保存至: {output_path}")
plt.show()






import matplotlib.font_manager as fm

# 获取当前图形的渲染器
fig = plt.gcf()
renderer = fig.canvas.get_renderer()

# 获取标题的文本对象
ax = plt.gca()
title_text = ax.title

# 获取文本的边界框和视觉信息，这会促使文本完成布局计算
title_text.get_window_extent(renderer)

# 然后通过字体管理器来查找匹配的字体
try:
    # 创建一个字体属性对象，基于文本的当前设置
    fp = title_text.get_fontproperties()
    # 让字体管理器为您解析并找到匹配的字体文件
    matched_font = fm.findfont(fp)
    print(f"字体管理器匹配到的字体文件: {matched_font}")
except Exception as e:
    print(f"通过字体管理器查找失败: {e}")













# ########第一bar
# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置中文字体

# # 设置全局字体大小（基准值，可被局部覆盖）
# rcParams.update({
#     'figure.dpi': 600,  # 降低 DPI 以避免显示问题
#     'font.size': 20,           # 默认字体大小（影响大多数文本）
#     'axes.titlesize': 20,      # 标题字体大小
#     'axes.labelsize': 20,      # 坐标轴标签大小
#     'xtick.labelsize': 20,      # X轴刻度标签大小
#     'ytick.labelsize': 20,      # Y轴刻度标签大小
#     'legend.fontsize': 20,     # 图例字体大小
# })
#
# # 指定存放 Excel 文件的文件夹路径
# folder_path = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/细分男女"  # 请替换为实际路径
#
# # 获取文件夹下所有 Excel 文件
# excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
#
# # 自定义顺序：根据文件名指定顺序
# custom_order = ["35-47", "48-59", "60-74", "G1", "G2", "G3+G4"]  # 替换为实际文件名（不带扩展名）
# excel_files = sorted(excel_files, key=lambda x: custom_order.index(os.path.splitext(x)[0]))
#
# # 初始化空列表存储数据
# data = []
#
# # 遍历所有 Excel 文件
# for file in excel_files:
#     file_path = os.path.join(folder_path, file)
#     group_name = os.path.splitext(file)[0]  # 以文件名作为组名
#
#     try:
#         # 读取 Excel 文件（假设前两列是类别数据）
#         df = pd.read_excel(file_path, usecols=[0, 1])  # 只读取前两列，避免格式问题
#
#         if df.shape[1] < 2:
#             print(f"⚠️ 文件 {file} 格式不符合要求，跳过")
#             continue
#
#         # 获取两列的列名（如 "COPD" 和 "Health"）
#         left_category, right_category = df.columns[:2]
#
#         # 转换数据格式（长格式）
#         for value in df[left_category].dropna():
#             data.append((group_name, "Male", value))  # 这里用统一的字符串
#         for value in df[right_category].dropna():
#             data.append((group_name, "Female", value))
#
#     except Exception as e:
#         print(f"❌ 处理文件 {file} 时出错：{e}")
#         continue
#
# # 转换为 DataFrame
# df_all = pd.DataFrame(data, columns=["Group", "Status", "Lung age gap"])
#
# # 确保 "Status" 列是字符串
# df_all["Status"] = df_all["Status"].astype(str)
#
# # 按照自定义顺序排序数据
# df_all["Group"] = pd.Categorical(df_all["Group"], categories=custom_order, ordered=True)
# df_all = df_all.sort_values("Group")
#
# # 生成 x 轴位置
# groups = df_all["Group"].unique()
# base_x_positions = list(range(len(groups)))
#
# # 生成映射：将分组名称映射到 x 轴位置
# group_to_position = {group: pos for group, pos in zip(groups, base_x_positions)}
#
# # 计算每组的均值
# group_means = df_all.groupby(["Group", "Status"])["Lung age gap"].mean().reset_index()
#
# # 画图
# plt.figure(figsize=(16, 8))
#
# # 颜色定义
# palette = {"Male": "#87CEEB", "Female": "#FFC0CB"}  # 确保颜色映射正确
#
# # 使用 seaborn 绘制箱线图
# sns.boxplot(x="Group", y="Lung age gap", hue="Status", data=df_all, palette=palette, width=0.6)
#
# # 添加参考线
# plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
#
# # 更新 x 轴刻度
# plt.xticks(ticks=range(len(custom_order)), labels=custom_order, rotation=0)
#
# # 隐藏 x 轴标签
# plt.xlabel("")  # 隐藏 "Group" 单词
# plt.ylim(-20,20)
# plt.yticks(ticks=[-20,  -15, -10, -5,     0,  5,   10,  15,  20],
#            labels=["-20","-15","-10", "-5",  "0", "5", "10", "15", "20"])
# # 添加图例
# plt.legend(title="")
#
# # 设置标题
# plt.title("")
#
# # 在每个分组上方添加均值文本
# for i, group in enumerate(custom_order):
#     # 获取 Male 和 Female 的均值
#     male_mean = group_means[(group_means["Group"] == group) & (group_means["Status"] == "Male")]["Lung age gap"].values
#     female_mean = group_means[(group_means["Group"] == group) & (group_means["Status"] == "Female")]["Lung age gap"].values
#
#     # 绘制 Male 均值文本
#     if len(male_mean) > 0:
#         plt.text(i - 0.15, male_mean[0] + 0.75, f"{male_mean[0]:.2f}",  # 调整文本位置和格式
#                  ha="center", va="bottom", fontsize=18, color="black")
#     # 绘制 Female 均值文本
#     if len(female_mean) > 0:
#         plt.text(i + 0.15, female_mean[0] + 0.75, f"{female_mean[0]:.2f}",  # 调整文本位置和格式
#                  ha="center", va="bottom", fontsize=18, color="black")
#
# # 定义每个分组的星号数量
# star_counts = {
#     "35-47": 3,   # 0-47 分组显示 1 个星号
#     "48-59": 1,  # 48-59 分组显示 3 个星号
#     "60-74": 0,  # 60-74 分组显示 3 个星号
#     "G1": 0,     # G1 分组显示 3 个星号
#     "G2": 0,     # G2 分组显示 3 个星号
#     "G3+G4": 0   # G3+G4 分组显示 3 个星号
# }
#
# # 定义每个星号的位置 (x, y)
# star_positions = {
#     "35-47": (0.24, -21.8),   # 0-47 分组的星号位置
#     "48-59": (1.19, -21.8),  # 48-59 分组的星号位置
#     "60-74": (2.17, -22.7),  # 60-74 分组的星号位置
#     "G1": (3.11, -22.7),     # G1 分组的星号位置
#     "G2": (4.11, -22.7),     # G2 分组的星号位置
#     "G3+G4": (5.2, -22.7)   # G3+G4 分组的星号位置
# }
#
# # 在每个分组标签的右上角添加星号
# for group, pos in group_to_position.items():
#     if group in star_counts and group in star_positions:
#         stars = "*" * star_counts[group]  # 根据字典生成星号
#         x, y = star_positions[group]  # 获取自定义的 x 和 y 坐标
#         plt.text(x, y, stars,  # 在指定位置绘制星号
#                  ha="center", va="bottom", fontsize=18, color="black")
#
# plt.tight_layout()
# plt.show()










#######之前小字
# ########PRISM,小提琴图
# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置中文字体

#
# # 指定存放 Excel 文件的文件夹路径
# folder_path = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/PRISM/绘图"  # 请替换为实际路径
#
# # 获取文件夹下所有 Excel 文件
# excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
#
# # 自定义顺序：根据文件名指定顺序
# custom_order = ["All", "Male", "Female", "35-47", "48-59", "60-74",  "G2", "G3+G4"]  # 替换为实际文件名（不带扩展名）
# excel_files = sorted(excel_files, key=lambda x: custom_order.index(os.path.splitext(x)[0]))
#
# # 初始化空列表存储数据
# data = []
#
# # 遍历所有 Excel 文件
# for file in excel_files:
#     file_path = os.path.join(folder_path, file)
#     group_name = os.path.splitext(file)[0]  # 以文件名作为组名
#
#     try:
#         # 读取 Excel 文件（假设前两列是类别数据）
#         df = pd.read_excel(file_path, usecols=[0, 1])  # 只读取前两列，避免格式问题
#
#         if df.shape[1] < 2:
#             print(f"⚠️ 文件 {file} 格式不符合要求，跳过")
#             continue
#
#         # 获取两列的列名（如 "COPD" 和 "Health"）
#         left_category, right_category = df.columns[:2]
#
#         # 转换数据格式（长格式）
#         for value in df[left_category].dropna():
#             data.append((group_name, "PRISM", value))  # 这里用统一的字符串
#         for value in df[right_category].dropna():
#             data.append((group_name, "Health", value))
#
#     except Exception as e:
#         print(f"❌ 处理文件 {file} 时出错：{e}")
#         continue
#
# # 转换为 DataFrame
# df_all = pd.DataFrame(data, columns=["Group", "Status", "Lung age gap"])
#
# # 确保 "Status" 列是字符串
# df_all["Status"] = df_all["Status"].astype(str)
#
# # 按照自定义顺序排序数据
# df_all["Group"] = pd.Categorical(df_all["Group"], categories=custom_order, ordered=True)
# df_all = df_all.sort_values("Group")
#
# # 生成 x 轴位置
# groups = df_all["Group"].unique()
# base_x_positions = list(range(len(groups)))
# x_offset = 0.15  # 控制左右小提琴的间距
#
# # 生成映射
# group_to_position = {group: pos for group, pos in zip(groups, base_x_positions)}
#
# # 计算新的 x 轴位置，使左右分开
# df_all["x_pos"] = df_all.apply(
#     lambda row: group_to_position[row["Group"]] - x_offset if row["Status"] == "PRISM"
#     else group_to_position[row["Group"]] + x_offset, axis=1)
#
# # 画图
# plt.figure(figsize=(12, 6))
#
# # 颜色定义
# palette = {"PRISM": "#FFFF99", "Health": "#FFC0CB"}  # 确保颜色映射正确
#
# # 使用 hue 直接区分类别，保证绘制正确
# sns.violinplot(x="Group", y="Lung age gap", hue="Status", data=df_all,
#                split=True, inner="quartile", palette=palette, linewidth=1, scale="width", width=0.6)  # 设置 scale="width"
#
# # 计算均值并添加标记
# group_means = df_all.groupby(["Group", "Status"])["Lung age gap"].mean().reset_index()
# for _, row in group_means.iterrows():
#     x_pos = group_to_position[row["Group"]] - x_offset if row["Status"] == "PRISM" else group_to_position[row["Group"]] + x_offset
#     plt.scatter(x_pos, row["Lung age gap"], color="black", marker="*", s=100, zorder=3)
#
#     # 在每个小提琴图的两半上方添加均值文本
#     plt.text(x_pos, row["Lung age gap"] + 0.5, f"{row['Lung age gap']:.2f}",  # 调整文本位置和格式
#              ha="center", va="bottom", fontsize=9, color="black")
#
# # 添加参考线
# plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
#
# # 更新 x 轴刻度
# plt.xticks(ticks=base_x_positions, labels=groups, rotation=0)
#
# # 隐藏 x 轴标签
# plt.xlabel("")  # 隐藏 "Group" 单词
#
# # 隐藏图例标题
# plt.legend(title="", loc='upper right')  # 隐藏 "Status" 单词
#
# # 设置标题
# plt.title("")
#
# # 定义每个分组的星号数量
# star_counts = {
#     "All": 2,    # All 分组显示 1 个星号
#     "Male": 1,   # Male 分组显示 2 个星号
#     "Female": 0, # Female 分组显示 3 个星号
#     "35-47": 1,   # 0-47 分组显示 1 个星号
#     "48-59": 2,  # 48-59 分组显示 2 个星号
#     "60-74": 0,  # 60-74 分组显示 3 个星号
#     "G2": 2,     # G2 分组显示 2 个星号
#     "G3+G4": 0   # G3+G4 分组显示 3 个星号
# }
#
# # 定义每个星号的位置 (x, y)
# star_positions = {
#     "All": (0.12, -29.5),    # All 分组的星号位置
#     "Male": (1.15, -29.5),   # Male 分组的星号位置
#     "Female": (2.3, -29.5), # Female 分组的星号位置
#     "35-47": (3.19, -29.5),   # 0-47 分组的星号位置
#     "48-59": (4.21, -29.5),  # 48-59 分组的星号位置
#     "60-74": (5.27, -29.5),   # 60-74 分组的星号位置
#     "G2": (6.15, -29.5),     # G2 分组的星号位置
#     "G3+G4": (8.31, -29.5)   # G3+G4 分组的星号位置
# }
#
# # 在每个分组标签的右上角添加星号
# for group, pos in group_to_position.items():
#     if group in star_counts and group in star_positions:
#         stars = "*" * star_counts[group]  # 根据字典生成星号
#         x, y = star_positions[group]  # 获取自定义的 x 和 y 坐标
#         plt.text(x, y, stars,  # 在指定位置绘制星号
#                  ha="center", va="bottom", fontsize=12, color="black")
#
#
# plt.tight_layout()
# plt.show()









##########PRISM,大字
# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置中文字体

#
#
# # 设置全局字体大小（基准值，可被局部覆盖）
# rcParams.update({
#     'figure.dpi': 600,  # 降低 DPI 以避免显示问题
#     'font.size': 20,           # 默认字体大小（影响大多数文本）
#     'axes.titlesize': 20,      # 标题字体大小
#     'axes.labelsize': 20,      # 坐标轴标签大小
#     'xtick.labelsize': 20,      # X轴刻度标签大小
#     'ytick.labelsize': 20,      # Y轴刻度标签大小
#     'legend.fontsize': 20,     # 图例字体大小
# })
#
# # 指定存放 Excel 文件的文件夹路径
# folder_path = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/PRISM/绘图"  # 请替换为实际路径
#
# # 获取文件夹下所有 Excel 文件
# excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
#
# # 自定义顺序：根据文件名指定顺序
# custom_order = ["All", "Male", "Female", "35-47", "48-59", "60-74", "G2", "G3+G4"]  # 替换为实际文件名（不带扩展名）
# excel_files = sorted(excel_files, key=lambda x: custom_order.index(os.path.splitext(x)[0]))
#
# # 初始化空列表存储数据
# data = []
#
# # 遍历所有 Excel 文件
# for file in excel_files:
#     file_path = os.path.join(folder_path, file)
#     group_name = os.path.splitext(file)[0]  # 以文件名作为组名
#
#     try:
#         # 读取 Excel 文件（假设前两列是类别数据）
#         df = pd.read_excel(file_path, usecols=[0, 1])  # 只读取前两列，避免格式问题
#
#         if df.shape[1] < 2:
#             print(f"⚠️ 文件 {file} 格式不符合要求，跳过")
#             continue
#
#         # 获取两列的列名（如 "COPD" 和 "Health"）
#         left_category, right_category = df.columns[:2]
#
#         # 转换数据格式（长格式）
#         for value in df[left_category].dropna():
#             data.append((group_name, "PRISm", value))  # 这里用统一的字符串
#         for value in df[right_category].dropna():
#             data.append((group_name, "Health", value))
#
#     except Exception as e:
#         print(f"❌ 处理文件 {file} 时出错：{e}")
#         continue
#
# # 转换为 DataFrame
# df_all = pd.DataFrame(data, columns=["Group", "Status", "Lung age gap"])
#
# # 确保 "Status" 列是字符串
# df_all["Status"] = df_all["Status"].astype(str)
#
# # 按照自定义顺序排序数据
# df_all["Group"] = pd.Categorical(df_all["Group"], categories=custom_order, ordered=True)
# df_all = df_all.sort_values("Group")
#
# # 生成 x 轴位置
# groups = df_all["Group"].unique()
# base_x_positions = list(range(len(groups)))
# x_offset = 0.15  # 控制左右小提琴的间距
#
# # 生成映射
# group_to_position = {group: pos for group, pos in zip(groups, base_x_positions)}
#
# # 计算新的 x 轴位置，使左右分开
# df_all["x_pos"] = df_all.apply(
#     lambda row: group_to_position[row["Group"]] - x_offset if row["Status"] == "PRISm"
#     else group_to_position[row["Group"]] + x_offset, axis=1)
#
# # 画图
# plt.figure(figsize=(16, 8))
# plt.ylim(-30,20)
# plt.yticks(ticks=[-30,  -24,  -18, -12,   -6,     0,   6,   12,   18,   24],
#            labels=["-25","-20","-15","-10", "-5",   "0", "5", "10", "15", "20"])
#
# # 颜色定义
# palette = {"PRISm": "#FFFF99", "Health": "#FFC0CB"}  # 确保颜色映射正确
#
# # 使用 hue 直接区分类别，保证绘制正确
# sns.violinplot(x="Group", y="Lung age gap", hue="Status", data=df_all,
#                split=True, inner="quartile", palette=palette, linewidth=1, scale="width", width=0.6)  # 设置 scale="width"
#
# # 计算均值并添加标记
# group_means = df_all.groupby(["Group", "Status"])["Lung age gap"].mean().reset_index()
#
# # 定义每个组的文本垂直和水平偏移量
# text_offsets = {
#     # 格式: {"Group": {"Status": (x_offset, y_offset)}}
#     "All": {"PRISm": (-0.1, 0.8), "Health": (0.1, 0.8)},
#     "Male": {"PRISm": (-0.1, 0.8), "Health": (0.1, 0.8)},
#     "Female": {"PRISm": (-0.1, 0.8), "Health": (0.1, 0.8)},
#     "35-47": {"PRISm": (-0.1, 0.7), "Health": (0.1, 0.7)},
#     "48-59": {"PRISm": (-0.1, 0.8), "Health": (0.1, 0.8)},
#     "60-74": {"PRISm": (-0.1, 0.9), "Health": (0.1, 0.9)},
#     "G2": {"PRISm": (-0.1, 0.8), "Health": (0.1, 0.8)},
#     "G3+G4": {"PRISm": (-0.1, 0.9), "Health": (0.1, 0.9)}
# }
#
# for _, row in group_means.iterrows():
#     base_x = group_to_position[row["Group"]]
#     x_pos = base_x - x_offset if row["Status"] == "PRISm" else base_x + x_offset
#
#     # 获取当前组的偏移量设置
#     offsets = text_offsets.get(row["Group"], {"PRISm": (0, 0), "Health": (0, 0)})
#     x_offset_text, y_offset_text = offsets[row["Status"]]
#
#     plt.scatter(x_pos, row["Lung age gap"], color="black", marker="*", s=100, zorder=3)
#
#     # 应用自定义偏移
#     plt.text(
#         x_pos + x_offset_text,  # 水平偏移
#         row["Lung age gap"] + y_offset_text,  # 垂直偏移
#         f"{row['Lung age gap']:.2f}",
#         ha="center",
#         va="bottom",
#         fontsize=16,
#         color="black",
#         bbox=dict(facecolor='none', alpha=0.7, pad=0.1, edgecolor='none')
#     )
#
#
# # 添加参考线
# plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
#
# # 更新 x 轴刻度
# plt.xticks(ticks=base_x_positions, labels=groups, rotation=0)
#
# # 隐藏 x 轴标签
# plt.xlabel("")  # 隐藏 "Group" 单词
#
# # 隐藏图例标题
# plt.legend(title="")  # 隐藏 "Status" 单词
#
# # 设置标题
# plt.title("")
#
# # 定义每个分组的星号数量
# star_counts = {
#     "All": 3,    # All 分组显示 1 个星号
#     "Male": 3,   # Male 分组显示 2 个星号
#     "Female": 3, # Female 分组显示 3 个星号
#     "35-47": 1,   # 0-47 分组显示 1 个星号
#     "48-59": 3,  # 48-59 分组显示 2 个星号
#     "60-74": 3,  # 60-74 分组显示 3 个星号
#     "G2": 3,     # G2 分组显示 2 个星号
#     "G3+G4": 3   # G3+G4 分组显示 3 个星号
# }
#
# # 定义每个星号的位置 (x, y)
# star_positions = {
#     "All": (0.21, -32.1),    # All 分组的星号位置
#     "Male": (1.30, -32.1),   # Male 分组的星号位置
#     "Female": (2.40, -32.1), # Female 分组的星号位置
#     "35-47": (3.27, -32.1),   # 0-47 分组的星号位置
#     "48-59": (4.35, -32.1),  # 48-59 分组的星号位置
#     "60-74": (5.35, -32.1),   # 60-74 分组的星号位置
#     "G2": (6.21, -32.1),     # G1 分组的星号位置
#     "G3+G4": (7.40, -32.1)     # G 分组的星号位置
#
# }
#
# # 在每个分组标签的右上角添加星号
# for group, pos in group_to_position.items():
#     if group in star_counts and group in star_positions:
#         stars = "*" * star_counts[group]  # 根据字典生成星号
#         x, y = star_positions[group]  # 获取自定义的 x 和 y 坐标
#         plt.text(x, y, stars,  # 在指定位置绘制星号
#                  ha="center", va="bottom", fontsize=18, color="black")
#
# plt.legend(title="", loc="upper right", bbox_to_anchor=(1, 1), frameon=True)
#
# plt.tight_layout()
# plt.show()







# ###########PRISM-柱状（小字）
# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置中文字体

#
# # 指定存放 Excel 文件的文件夹路径
# folder_path = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/PRISM/抽取之后的男女"  # 请替换为实际路径
#
# # 获取文件夹下所有 Excel 文件
# excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
#
# # 自定义顺序：根据文件名指定顺序
# custom_order = ["35-47", "48-59", "60-74",  "G2", "G3+G4"]  # 替换为实际文件名（不带扩展名）
# excel_files = sorted(excel_files, key=lambda x: custom_order.index(os.path.splitext(x)[0]))
#
# # 初始化空列表存储数据
# data = []
#
# # 遍历所有 Excel 文件
# for file in excel_files:
#     file_path = os.path.join(folder_path, file)
#     group_name = os.path.splitext(file)[0]  # 以文件名作为组名
#
#     try:
#         # 读取 Excel 文件（假设前两列是类别数据）
#         df = pd.read_excel(file_path, usecols=[0, 1])  # 只读取前两列，避免格式问题
#
#         if df.shape[1] < 2:
#             print(f"⚠️ 文件 {file} 格式不符合要求，跳过")
#             continue
#
#         # 获取两列的列名（如 "COPD" 和 "Health"）
#         left_category, right_category = df.columns[:2]
#
#         # 转换数据格式（长格式）
#         for value in df[left_category].dropna():
#             data.append((group_name, "Male", value))  # 这里用统一的字符串
#         for value in df[right_category].dropna():
#             data.append((group_name, "Female", value))
#
#     except Exception as e:
#         print(f"❌ 处理文件 {file} 时出错：{e}")
#         continue
#
# # 转换为 DataFrame
# df_all = pd.DataFrame(data, columns=["Group", "Status", "Lung age gap"])
#
# # 确保 "Status" 列是字符串
# df_all["Status"] = df_all["Status"].astype(str)
#
# # 按照自定义顺序排序数据
# df_all["Group"] = pd.Categorical(df_all["Group"], categories=custom_order, ordered=True)
# df_all = df_all.sort_values("Group")
#
# # 生成 x 轴位置
# groups = df_all["Group"].unique()
# base_x_positions = list(range(len(groups)))
#
# # 生成映射：将分组名称映射到 x 轴位置
# group_to_position = {group: pos for group, pos in zip(groups, base_x_positions)}
#
# # 计算每组的均值
# group_means = df_all.groupby(["Group", "Status"])["Lung age gap"].mean().reset_index()
#
# # 画图
# plt.figure(figsize=(12, 6))
#
# # 颜色定义
# palette = {"Male": "#FFFF99", "Female": "#FFC0CB"}  # 确保颜色映射正确
#
# # 使用 seaborn 绘制箱线图
# sns.boxplot(x="Group", y="Lung age gap", hue="Status", data=df_all, palette=palette, width=0.6)
#
# # 添加参考线
# plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
#
# # 更新 x 轴刻度
# plt.xticks(ticks=range(len(custom_order)), labels=custom_order, rotation=0)
#
# # 隐藏 x 轴标签
# plt.xlabel("")  # 隐藏 "Group" 单词
#
# # 添加图例
# plt.legend(title="")
#
# # 设置标题
# plt.title("")
#
# # 在每个分组上方添加均值文本
# for i, group in enumerate(custom_order):
#     # 获取 Male 和 Female 的均值
#     male_mean = group_means[(group_means["Group"] == group) & (group_means["Status"] == "Male")]["Lung age gap"].values
#     female_mean = group_means[(group_means["Group"] == group) & (group_means["Status"] == "Female")]["Lung age gap"].values
#
#     # 绘制 Male 均值文本
#     if len(male_mean) > 0:
#         plt.text(i - 0.15, male_mean[0] + 0.75, f"{male_mean[0]:.2f}",  # 调整文本位置和格式
#                  ha="center", va="bottom", fontsize=9, color="black")
#     # 绘制 Female 均值文本
#     if len(female_mean) > 0:
#         plt.text(i + 0.15, female_mean[0] + 0.75, f"{female_mean[0]:.2f}",  # 调整文本位置和格式
#                  ha="center", va="bottom", fontsize=9, color="black")
#
# # 定义每个分组的星号数量
# star_counts = {
#     "35-47": 0,   # 0-47 分组显示 1 个星号
#     "48-59": 0,  # 48-59 分组显示 3 个星号
#     "60-74": 0,  # 60-74 分组显示 3 个星号
#     "G2": 1,     # G2 分组显示 3 个星号
#     "G3+G4": 0   # G3+G4 分组显示 3 个星号
# }
#
# # 定义每个星号的位置 (x, y)
# star_positions = {
#     "35-47": (0.145, -25.7),   # 0-47 分组的星号位置
#     "48-59": (1.1, -25.7),  # 48-59 分组的星号位置
#     "60-74": (2.17, -19.7),  # 60-74 分组的星号位置
#     "G2": (3.07, -25.7),     # G2 分组的星号位置
#     "G3+G4": (5.2, -19.7)   # G3+G4 分组的星号位置
# }
#
# # 在每个分组标签的右上角添加星号
# for group, pos in group_to_position.items():
#     if group in star_counts and group in star_positions:
#         stars = "*" * star_counts[group]  # 根据字典生成星号
#         x, y = star_positions[group]  # 获取自定义的 x 和 y 坐标
#         plt.text(x, y, stars,  # 在指定位置绘制星号
#                  ha="center", va="bottom", fontsize=12, color="black")
#
# plt.tight_layout()
# plt.show()






# ###########PRISm（bar大字）
# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置中文字体

# # 设置全局字体大小（基准值，可被局部覆盖）
# rcParams.update({
#     'figure.dpi': 600,  # 降低 DPI 以避免显示问题
#     'font.size': 20,           # 默认字体大小（影响大多数文本）
#     'axes.titlesize': 20,      # 标题字体大小
#     'axes.labelsize': 20,      # 坐标轴标签大小
#     'xtick.labelsize': 20,      # X轴刻度标签大小
#     'ytick.labelsize': 20,      # Y轴刻度标签大小
#     'legend.fontsize': 20,     # 图例字体大小
# })
#
# # 指定存放 Excel 文件的文件夹路径
# folder_path = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/PRISM/抽取之后的男女"  # 请替换为实际路径
#
# # 获取文件夹下所有 Excel 文件
# excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
#
# # 自定义顺序：根据文件名指定顺序
# custom_order = ["35-47", "48-59", "60-74", "G2", "G3+G4"]  # 替换为实际文件名（不带扩展名）
# excel_files = sorted(excel_files, key=lambda x: custom_order.index(os.path.splitext(x)[0]))
#
# # 初始化空列表存储数据
# data = []
#
# # 遍历所有 Excel 文件
# for file in excel_files:
#     file_path = os.path.join(folder_path, file)
#     group_name = os.path.splitext(file)[0]  # 以文件名作为组名
#
#     try:
#         # 读取 Excel 文件（假设前两列是类别数据）
#         df = pd.read_excel(file_path, usecols=[0, 1])  # 只读取前两列，避免格式问题
#
#         if df.shape[1] < 2:
#             print(f"⚠️ 文件 {file} 格式不符合要求，跳过")
#             continue
#
#         # 获取两列的列名（如 "COPD" 和 "Health"）
#         left_category, right_category = df.columns[:2]
#
#         # 转换数据格式（长格式）
#         for value in df[left_category].dropna():
#             data.append((group_name, "Male", value))  # 这里用统一的字符串
#         for value in df[right_category].dropna():
#             data.append((group_name, "Female", value))
#
#     except Exception as e:
#         print(f"❌ 处理文件 {file} 时出错：{e}")
#         continue
#
# # 转换为 DataFrame
# df_all = pd.DataFrame(data, columns=["Group", "Status", "Lung age gap"])
#
# # 确保 "Status" 列是字符串
# df_all["Status"] = df_all["Status"].astype(str)
#
# # 按照自定义顺序排序数据
# df_all["Group"] = pd.Categorical(df_all["Group"], categories=custom_order, ordered=True)
# df_all = df_all.sort_values("Group")
#
# # 生成 x 轴位置
# groups = df_all["Group"].unique()
# base_x_positions = list(range(len(groups)))
#
# # 生成映射：将分组名称映射到 x 轴位置
# group_to_position = {group: pos for group, pos in zip(groups, base_x_positions)}
#
# # 计算每组的均值
# group_means = df_all.groupby(["Group", "Status"])["Lung age gap"].mean().reset_index()
#
# # 画图
# plt.figure(figsize=(16, 8))
# plt.ylim(-25,20)
#
# plt.yticks(ticks=[-25,  -20, -15,    -10,   -5,      0,  5,  10,  15,  20],
#            labels=["-25","-20","-15","-10", "-5",   "0", "5", "10", "15", "20"])
#
# # 颜色定义
# palette = {"Male": "#FFFF99", "Female": "#FFC0CB"}  # 确保颜色映射正确
#
# # 使用 seaborn 绘制箱线图
# sns.boxplot(x="Group", y="Lung age gap", hue="Status", data=df_all, palette=palette, width=0.6)
#
# # 添加参考线
# plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
#
# # 更新 x 轴刻度
# plt.xticks(ticks=range(len(custom_order)), labels=custom_order, rotation=0)
#
# # 隐藏 x 轴标签
# plt.xlabel("")  # 隐藏 "Group" 单词
#
# # 添加图例
# plt.legend(title="")
#
# # 设置标题
# plt.title("")
#
# # 在每个分组上方添加均值文本
# for i, group in enumerate(custom_order):
#     # 获取 Male 和 Female 的均值
#     male_mean = group_means[(group_means["Group"] == group) & (group_means["Status"] == "Male")]["Lung age gap"].values
#     female_mean = group_means[(group_means["Group"] == group) & (group_means["Status"] == "Female")]["Lung age gap"].values
#
#     # 绘制 Male 均值文本
#     if len(male_mean) > 0:
#         plt.text(i - 0.15, male_mean[0] + 0.75, f"{male_mean[0]:.2f}",  # 调整文本位置和格式
#                  ha="center", va="bottom", fontsize=18, color="black")
#     # 绘制 Female 均值文本
#     if len(female_mean) > 0:
#         plt.text(i + 0.15, female_mean[0] + 0.75, f"{female_mean[0]:.2f}",  # 调整文本位置和格式
#                  ha="center", va="bottom", fontsize=18, color="black")
#
# # 定义每个分组的星号数量
# star_counts = {
#     "35-47": 0,   # 0-47 分组显示 1 个星号
#     "48-59": 0,  # 48-59 分组显示 3 个星号
#     "60-74": 0,  # 60-74 分组显示 3 个星号
#     "G2": 1,     # G2 分组显示 3 个星号
#     "G3+G4": 0   # G3+G4 分组显示 3 个星号
# }
#
# # 定义每个星号的位置 (x, y)
# star_positions = {
#     "35-47": (0.145, -25.7),   # 0-47 分组的星号位置
#     "48-59": (1.1, -25.7),  # 48-59 分组的星号位置
#     "60-74": (2.17, -19.7),  # 60-74 分组的星号位置
#     "G2": (3.08, -26.7),     # G2 分组的星号位置
#     "G3+G4": (5.2, -19.7)   # G3+G4 分组的星号位置
# }
#
# # 在每个分组标签的右上角添加星号
# for group, pos in group_to_position.items():
#     if group in star_counts and group in star_positions:
#         stars = "*" * star_counts[group]  # 根据字典生成星号
#         x, y = star_positions[group]  # 获取自定义的 x 和 y 坐标
#         plt.text(x, y, stars,  # 在指定位置绘制星号
#                  ha="center", va="bottom", fontsize=18, color="black")
#
# plt.tight_layout()
# plt.show()








# ########分年龄段###############小提琴图2
# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置中文字体

#
# # 指定存放 Excel 文件的文件夹路径
# folder_path = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/分年龄段"  # 请替换为实际路径
#
# # 获取文件夹下所有 Excel 文件
# excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
#
# # 初始化空列表存储数据
# data = []
#
# # 遍历所有 Excel 文件
# for file in excel_files:
#     file_path = os.path.join(folder_path, file)
#     group_name = os.path.splitext(file)[0]  # 以文件名作为组名
#
#     try:
#         # 读取 Excel 文件（假设前两列是类别数据）
#         df = pd.read_excel(file_path, usecols=[0, 1])  # 只读取前两列，避免格式问题
#
#         if df.shape[1] < 2:
#             print(f"⚠️ 文件 {file} 格式不符合要求，跳过")
#             continue
#
#         # 获取两列的列名（如 "COPD" 和 "Health"）
#         left_category, right_category = df.columns[:2]
#
#         # 转换数据格式（长格式）
#         for value in df[left_category].dropna():
#             data.append((group_name, "COPD", value))  # 这里用统一的字符串
#         for value in df[right_category].dropna():
#             data.append((group_name, "Health", value))
#
#     except Exception as e:
#         print(f"❌ 处理文件 {file} 时出错：{e}")
#         continue
#
# # 转换为 DataFrame
# df_all = pd.DataFrame(data, columns=["Group", "Status", "Body Age Gap"])
#
# # 确保 "Status" 列是字符串
# df_all["Status"] = df_all["Status"].astype(str)
#
# # 生成 x 轴位置
# groups = df_all["Group"].unique()
# base_x_positions = list(range(len(groups)))
# x_offset = 0.15  # 控制左右小提琴的间距
#
# # 生成映射
# group_to_position = {group: pos for group, pos in zip(groups, base_x_positions)}
#
# # 计算新的 x 轴位置，使左右分开
# df_all["x_pos"] = df_all.apply(
#     lambda row: group_to_position[row["Group"]] - x_offset if row["Status"] == "COPD"
#     else group_to_position[row["Group"]] + x_offset, axis=1)
#
# # 画图
# plt.figure(figsize=(12, 6))
#
# # 颜色定义
# palette = {"COPD": "#87CEEB", "Health": "#FFC0CB"}  # 确保颜色映射正确
#
# # 使用 hue 直接区分类别，保证绘制正确
# sns.violinplot(x="Group", y="Body Age Gap", hue="Status", data=df_all,
#                split=True, inner="quartile", palette=palette, linewidth=1)
#
# # 计算均值并添加标记
# group_means = df_all.groupby(["Group", "Status"])["Body Age Gap"].mean().reset_index()
# for _, row in group_means.iterrows():
#     x_pos = group_to_position[row["Group"]] - x_offset if row["Status"] == "COPD" else group_to_position[row["Group"]] + x_offset
#     plt.scatter(x_pos, row["Body Age Gap"], color="black", marker="*", s=100, zorder=3)
#
# # 添加参考线
# plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
#
# # 更新 x 轴刻度
# plt.xticks(ticks=base_x_positions, labels=groups, rotation=30)
#
# # 设置标题
# plt.title("Body Age Gap by Group and Status")
# from matplotlib.patches import Patch
#
# # # 先创建第一个图例
# # legend1 = plt.legend(title="Status", bbox_to_anchor=(1, 1), loc="upper left")
# #
# # # 添加第一个图例到当前绘图区
# # plt.gca().add_artist(legend1)  # 这样第一个图例不会被覆盖
# #
# # # 第二个图例
# # extra_legend = [Patch(color="#87CEEB", label="Extra Type A"),
# #                 Patch(color="#FFC0CB", label="Extra Type B")]
# #
# # # 添加第二个图例到不同位置
# # plt.legend(handles=extra_legend, bbox_to_anchor=(0.2, 0.8), loc="upper left")
#
# from matplotlib.patches import Patch
#
# from matplotlib.patches import Patch
#
# # 先创建第一个图例 (自定义颜色)
# legend1_handles = [Patch(color="#87CEEB", label="mean=3.24"),
#                    Patch(color="#FFC0CB", label="mean=2.28")]
#
# # 生成第一个图例
# legend1 = plt.legend(handles=legend1_handles, title="", bbox_to_anchor=(0.05, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend1)
#
# # 创建第二个图例 (自定义颜色)
# legend2_handles = [Patch(color="#87CEEB", label="mean= 1.73"),
#                    Patch(color="#FFC0CB", label="mean=-0.43")]
#
# legend2 = plt.legend(handles=legend2_handles, title="", bbox_to_anchor=(0.27, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend2)
#
# # 创建第三个图例 (自定义颜色)
# legend3_handles = [Patch(color="#87CEEB", label="mean=-1.00"),
#                    Patch(color="#FFC0CB", label="mean=-3.53")]
#
# legend3 = plt.legend(handles=legend3_handles, title="", bbox_to_anchor=(0.51, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend3)
#
#
# # 创建第四个图例 (自定义颜色)
# legend3_handles = [Patch(color="#87CEEB", label="mean= 1.33"),
#                    Patch(color="#FFC0CB", label="mean=-0.64")]
#
# legend3 = plt.legend(handles=legend3_handles, title="", bbox_to_anchor=(0.75, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend3)
#
#
#
# # 生成第五个图例
# extra_legend_handles = [Patch(color="#87CEEB", label="COPD"),
#                         Patch(color="#FFC0CB", label="Health")]
#
# plt.legend(handles=extra_legend_handles, title="Status", bbox_to_anchor=(1, 1), loc="upper left",
#            prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
#
#
# plt.tight_layout()
# plt.show()
#











# #######COPD严重程度################小提琴图3
# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# import matplotlib.pyplot as plt
#

#
# # 指定存放 Excel 文件的文件夹路径
# folder_path = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/COPD严重程度/绘图严重程度"  # 请替换为实际路径
#
# # 获取文件夹下所有 Excel 文件
# excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
#
# # 初始化空列表存储数据
# data = []
#
# # 遍历所有 Excel 文件
# for file in excel_files:
#     file_path = os.path.join(folder_path, file)
#     group_name = os.path.splitext(file)[0]  # 以文件名作为组名
#
#     try:
#         # 读取 Excel 文件（假设前两列是类别数据）
#         df = pd.read_excel(file_path, usecols=[0, 1])  # 只读取前两列，避免格式问题
#
#         if df.shape[1] < 2:
#             print(f"⚠️ 文件 {file} 格式不符合要求，跳过")
#             continue
#
#         # 获取两列的列名（如 "COPD" 和 "Health"）
#         left_category, right_category = df.columns[:2]
#
#         # 转换数据格式（长格式）
#         for value in df[left_category].dropna():
#             data.append((group_name, "COPD", value))  # 这里用统一的字符串
#         for value in df[right_category].dropna():
#             data.append((group_name, "Health", value))
#
#     except Exception as e:
#         print(f"❌ 处理文件 {file} 时出错：{e}")
#         continue
#
# # 转换为 DataFrame
# df_all = pd.DataFrame(data, columns=["Group", "Status", "Body Age Gap"])
#
# # 确保 "Status" 列是字符串
# df_all["Status"] = df_all["Status"].astype(str)
#
# # 生成 x 轴位置
# groups = df_all["Group"].unique()
# base_x_positions = list(range(len(groups)))
# x_offset = 0.15  # 控制左右小提琴的间距
#
# # 生成映射
# group_to_position = {group: pos for group, pos in zip(groups, base_x_positions)}
#
# # 计算新的 x 轴位置，使左右分开
# df_all["x_pos"] = df_all.apply(
#     lambda row: group_to_position[row["Group"]] - x_offset if row["Status"] == "COPD"
#     else group_to_position[row["Group"]] + x_offset, axis=1)
#
# # 画图
# plt.figure(figsize=(12, 6))
#
# # 颜色定义
# palette = {"COPD": "#87CEEB", "Health": "#FFC0CB"}  # 确保颜色映射正确
#
# # 使用 hue 直接区分类别，保证绘制正确
# sns.violinplot(x="Group", y="Body Age Gap", hue="Status", data=df_all,
#                split=True, inner="quartile", palette=palette, linewidth=1)
#
# # 计算均值并添加标记
# group_means = df_all.groupby(["Group", "Status"])["Body Age Gap"].mean().reset_index()
# for _, row in group_means.iterrows():
#     x_pos = group_to_position[row["Group"]] - x_offset if row["Status"] == "COPD" else group_to_position[row["Group"]] + x_offset
#     plt.scatter(x_pos, row["Body Age Gap"], color="black", marker="*", s=100, zorder=3)
#
# # 添加参考线
# plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
#
# # 更新 x 轴刻度
# plt.xticks(ticks=base_x_positions, labels=groups, rotation=30)
#
# # 设置标题
# plt.title("Body Age Gap by Group and Status")
# from matplotlib.patches import Patch
#
# # # 先创建第一个图例
# # legend1 = plt.legend(title="Status", bbox_to_anchor=(1, 1), loc="upper left")
# #
# # # 添加第一个图例到当前绘图区
# # plt.gca().add_artist(legend1)  # 这样第一个图例不会被覆盖
# #
# # # 第二个图例
# # extra_legend = [Patch(color="#87CEEB", label="Extra Type A"),
# #                 Patch(color="#FFC0CB", label="Extra Type B")]
# #
# # # 添加第二个图例到不同位置
# # plt.legend(handles=extra_legend, bbox_to_anchor=(0.2, 0.8), loc="upper left")
#
# from matplotlib.patches import Patch
#
# from matplotlib.patches import Patch
#
# # 先创建第一个图例 (自定义颜色)
# legend1_handles = [Patch(color="#87CEEB", label="mean= 1.22"),
#                    Patch(color="#FFC0CB", label="mean=-0.66")]
#
# # 生成第一个图例
# legend1 = plt.legend(handles=legend1_handles, title="", bbox_to_anchor=(0.07, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend1)
#
# # 创建第二个图例 (自定义颜色)
# legend2_handles = [Patch(color="#87CEEB", label="mean= 1.40"),
#                    Patch(color="#FFC0CB", label="mean=-0.68")]
#
# legend2 = plt.legend(handles=legend2_handles, title="", bbox_to_anchor=(0.40, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend2)
#
#
#
# # 创建第三个图例 (自定义颜色)
# legend3_handles = [Patch(color="#87CEEB", label="mean= 1.36"),
#                    Patch(color="#FFC0CB", label="mean=-1.43")]
#
# legend3 = plt.legend(handles=legend3_handles, title="", bbox_to_anchor=(0.72, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend3)
#
#
#
# # 生成第五个图例
# extra_legend_handles = [Patch(color="#87CEEB", label="COPD"),
#                         Patch(color="#FFC0CB", label="Health")]
#
# plt.legend(handles=extra_legend_handles, title="Status", bbox_to_anchor=(1, 1), loc="upper left",
#            prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
#
#
# plt.tight_layout()
# plt.show()








# #######COPD严重程度分男女################小提琴图4
# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# import matplotlib.pyplot as plt
#

#
# # 指定存放 Excel 文件的文件夹路径
# folder_path = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/COPD严重程度/COPD各阶段分男女/COPD各阶段分男女绘图"  # 请替换为实际路径
#
# # 获取文件夹下所有 Excel 文件
# excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
#
# # 初始化空列表存储数据
# data = []
#
# # 遍历所有 Excel 文件
# for file in excel_files:
#     file_path = os.path.join(folder_path, file)
#     group_name = os.path.splitext(file)[0]  # 以文件名作为组名
#
#     try:
#         # 读取 Excel 文件（假设前两列是类别数据）
#         df = pd.read_excel(file_path, usecols=[0, 1])  # 只读取前两列，避免格式问题
#
#         if df.shape[1] < 2:
#             print(f"⚠️ 文件 {file} 格式不符合要求，跳过")
#             continue
#
#         # 获取两列的列名（如 "COPD" 和 "Health"）
#         left_category, right_category = df.columns[:2]
#
#         # 转换数据格式（长格式）
#         for value in df[left_category].dropna():
#             data.append((group_name, "COPD", value))  # 这里用统一的字符串
#         for value in df[right_category].dropna():
#             data.append((group_name, "Health", value))
#
#     except Exception as e:
#         print(f"❌ 处理文件 {file} 时出错：{e}")
#         continue
#
# # 转换为 DataFrame
# df_all = pd.DataFrame(data, columns=["Group", "Status", "Body Age Gap"])
#
# # 确保 "Status" 列是字符串
# df_all["Status"] = df_all["Status"].astype(str)
#
# # 生成 x 轴位置
# groups = df_all["Group"].unique()
# base_x_positions = list(range(len(groups)))
# x_offset = 0.15  # 控制左右小提琴的间距
#
# # 生成映射
# group_to_position = {group: pos for group, pos in zip(groups, base_x_positions)}
#
# # 计算新的 x 轴位置，使左右分开
# df_all["x_pos"] = df_all.apply(
#     lambda row: group_to_position[row["Group"]] - x_offset if row["Status"] == "COPD"
#     else group_to_position[row["Group"]] + x_offset, axis=1)
#
# # 画图
# plt.figure(figsize=(12, 6))
#
# # 颜色定义
# palette = {"COPD": "#87CEEB", "Health": "#FFC0CB"}  # 确保颜色映射正确
#
# # 使用 hue 直接区分类别，保证绘制正确
# sns.violinplot(x="Group", y="Body Age Gap", hue="Status", data=df_all,
#                split=True, inner="quartile", palette=palette, linewidth=1)
#
# # 计算均值并添加标记
# group_means = df_all.groupby(["Group", "Status"])["Body Age Gap"].mean().reset_index()
# for _, row in group_means.iterrows():
#     x_pos = group_to_position[row["Group"]] - x_offset if row["Status"] == "COPD" else group_to_position[row["Group"]] + x_offset
#     plt.scatter(x_pos, row["Body Age Gap"], color="black", marker="*", s=100, zorder=3)
#
# # 添加参考线
# plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
#
# # 更新 x 轴刻度
# plt.xticks(ticks=base_x_positions, labels=groups, rotation=30)
#
# # 设置标题
# plt.title("Body Age Gap by Group and Status")
# from matplotlib.patches import Patch
#
# # # 先创建第一个图例
# # legend1 = plt.legend(title="Status", bbox_to_anchor=(1, 1), loc="upper left")
# #
# # # 添加第一个图例到当前绘图区
# # plt.gca().add_artist(legend1)  # 这样第一个图例不会被覆盖
# #
# # # 第二个图例
# # extra_legend = [Patch(color="#87CEEB", label="Extra Type A"),
# #                 Patch(color="#FFC0CB", label="Extra Type B")]
# #
# # # 添加第二个图例到不同位置
# # plt.legend(handles=extra_legend, bbox_to_anchor=(0.2, 0.8), loc="upper left")
#
# from matplotlib.patches import Patch
#
# from matplotlib.patches import Patch
#
# # 先创建第一个图例 (自定义颜色)
# legend1_handles = [Patch(color="#87CEEB", label="mean= 2.15"),
#                    Patch(color="#FFC0CB", label="mean=-0.53")]
#
# # 生成第一个图例
# legend1 = plt.legend(handles=legend1_handles, title="", bbox_to_anchor=(0.07, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend1)
#
# # 创建第二个图例 (自定义颜色)
# legend2_handles = [Patch(color="#87CEEB", label="mean= 1.70"),
#                    Patch(color="#FFC0CB", label="mean= 0.93")]
#
# legend2 = plt.legend(handles=legend2_handles, title="", bbox_to_anchor=(0.40, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend2)
#
#
#
# # 创建第三个图例 (自定义颜色)
# legend3_handles = [Patch(color="#87CEEB", label="mean= 1.60"),
#                    Patch(color="#FFC0CB", label="mean= 0.69")]
#
# legend3 = plt.legend(handles=legend3_handles, title="", bbox_to_anchor=(0.72, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend3)
#
#
#
# # 生成第四个图例
# extra_legend_handles = [Patch(color="#87CEEB", label="Male"),
#                         Patch(color="#FFC0CB", label="Female")]
#
# plt.legend(handles=extra_legend_handles, title="Status", bbox_to_anchor=(1, 1), loc="upper left",
#            prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
#
#
# plt.tight_layout()
# plt.show()







#
# #######COPD年龄段分男女################小提琴图5
# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# import matplotlib.pyplot as plt
#

#
# # 指定存放 Excel 文件的文件夹路径
# folder_path = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/分年龄段/年龄段分男女/COPD年龄段分男女绘图"  # 请替换为实际路径
#
# # 获取文件夹下所有 Excel 文件
# excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
#
# # 初始化空列表存储数据
# data = []
#
# # 遍历所有 Excel 文件
# for file in excel_files:
#     file_path = os.path.join(folder_path, file)
#     group_name = os.path.splitext(file)[0]  # 以文件名作为组名
#
#     try:
#         # 读取 Excel 文件（假设前两列是类别数据）
#         df = pd.read_excel(file_path, usecols=[0, 1])  # 只读取前两列，避免格式问题
#
#         if df.shape[1] < 2:
#             print(f"⚠️ 文件 {file} 格式不符合要求，跳过")
#             continue
#
#         # 获取两列的列名（如 "COPD" 和 "Health"）
#         left_category, right_category = df.columns[:2]
#
#         # 转换数据格式（长格式）
#         for value in df[left_category].dropna():
#             data.append((group_name, "COPD", value))  # 这里用统一的字符串
#         for value in df[right_category].dropna():
#             data.append((group_name, "Health", value))
#
#     except Exception as e:
#         print(f"❌ 处理文件 {file} 时出错：{e}")
#         continue
#
# # 转换为 DataFrame
# df_all = pd.DataFrame(data, columns=["Group", "Status", "Body Age Gap"])
#
# # 确保 "Status" 列是字符串
# df_all["Status"] = df_all["Status"].astype(str)
#
# # 生成 x 轴位置
# groups = df_all["Group"].unique()
# base_x_positions = list(range(len(groups)))
# x_offset = 0.15  # 控制左右小提琴的间距
#
# # 生成映射
# group_to_position = {group: pos for group, pos in zip(groups, base_x_positions)}
#
# # 计算新的 x 轴位置，使左右分开
# df_all["x_pos"] = df_all.apply(
#     lambda row: group_to_position[row["Group"]] - x_offset if row["Status"] == "COPD"
#     else group_to_position[row["Group"]] + x_offset, axis=1)
#
# # 画图
# plt.figure(figsize=(12, 6))
#
# # 颜色定义
# palette = {"COPD": "#87CEEB", "Health": "#FFC0CB"}  # 确保颜色映射正确
#
# # 使用 hue 直接区分类别，保证绘制正确
# sns.violinplot(x="Group", y="Body Age Gap", hue="Status", data=df_all,
#                split=True, inner="quartile", palette=palette, linewidth=1)
#
# # 计算均值并添加标记
# group_means = df_all.groupby(["Group", "Status"])["Body Age Gap"].mean().reset_index()
# for _, row in group_means.iterrows():
#     x_pos = group_to_position[row["Group"]] - x_offset if row["Status"] == "COPD" else group_to_position[row["Group"]] + x_offset
#     plt.scatter(x_pos, row["Body Age Gap"], color="black", marker="*", s=100, zorder=3)
#
# # 添加参考线
# plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
#
# # 更新 x 轴刻度
# plt.xticks(ticks=base_x_positions, labels=groups, rotation=30)
#
# # 设置标题
# plt.title("Body Age Gap by Group and Status")
from matplotlib.patches import Patch

# # 先创建第一个图例
# legend1 = plt.legend(title="Status", bbox_to_anchor=(1, 1), loc="upper left")
#
# # 添加第一个图例到当前绘图区
# plt.gca().add_artist(legend1)  # 这样第一个图例不会被覆盖
#
# # 第二个图例
# extra_legend = [Patch(color="#87CEEB", label="Extra Type A"),
#                 Patch(color="#FFC0CB", label="Extra Type B")]
#
# # 添加第二个图例到不同位置
# plt.legend(handles=extra_legend, bbox_to_anchor=(0.2, 0.8), loc="upper left")
#
# from matplotlib.patches import Patch
#
# from matplotlib.patches import Patch
#
# # 先创建第一个图例 (自定义颜色)
# legend1_handles = [Patch(color="#87CEEB", label="mean= 4.39"),
#                    Patch(color="#FFC0CB", label="mean= 1.89")]
#
# # 生成第一个图例
# legend1 = plt.legend(handles=legend1_handles, title="", bbox_to_anchor=(0.07, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend1)
#
# # 创建第二个图例 (自定义颜色)
# legend2_handles = [Patch(color="#87CEEB", label="mean= 2.17"),
#                    Patch(color="#FFC0CB", label="mean= 0.92")]
#
# legend2 = plt.legend(handles=legend2_handles, title="", bbox_to_anchor=(0.40, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend2)
#
#
#
# # 创建第三个图例 (自定义颜色)
# legend3_handles = [Patch(color="#87CEEB", label="mean=-1.25"),
#                    Patch(color="#FFC0CB", label="mean=-0.62")]
#
# legend3 = plt.legend(handles=legend3_handles, title="", bbox_to_anchor=(0.74, 1), loc="upper left",
#                      prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
# plt.gca().add_artist(legend3)
#
#
#
# # 生成第四个图例
# extra_legend_handles = [Patch(color="#87CEEB", label="Male"),
#                         Patch(color="#FFC0CB", label="Female")]
#
# plt.legend(handles=extra_legend_handles, title="Status", bbox_to_anchor=(1, 1), loc="upper left",
#            prop={'size': 13}, markerscale=1.5)  # ✅ 放大字体 & 标记
#
#
# plt.tight_layout()
# plt.show()
#
#









# ############散点图1   FEV1--偏差
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr
#
# def average_every_n(data, n=2):
#     """接收一个 Series 并返回一个新的 Series，其中每 n 个元素的平均值"""
#     return data.groupby(data.index // n).mean()
#
# # 读取 Excel 文件
# file_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/散点图FEV1_偏差/FEV1_Pred——偏差/COPD严重程度.xlsx'  # 替换为你的文件路径
# data = pd.read_excel(file_path)
#
# # 假设第一列是 y（纵轴数据），第二列是 x（横轴数据）
# y = data.iloc[:, 0]  # 第一列
# x = data.iloc[:, 1]  # 第二列
#
# # 设置 n 的值
# n = 1  # 你可以根据需要更改这个值
#
# # 对数据进行处理：每 n 个数据取平均值
# y_avg = average_every_n(y, n)
# x_avg = average_every_n(x, n)
#
# # 计算 Pearson 相关系数 r 和 P 值
# r, p_value = pearsonr(x_avg, y_avg)
#
# # 创建图表
# plt.figure(figsize=(6,6))
# sns.kdeplot(x=x_avg, y=y_avg, cmap="rainbow", fill=True)
# plt.scatter(x_avg, y_avg, color='blue', alpha=0.5, s=10)
#
# # 1:1 参考线
# # plt.plot([min(x_avg), max(x_avg)], [min(x_avg), max(x_avg)], 'k--', lw=2)
#
# # 设定标题和标签
# plt.xlabel("Body Age Gap")  # 如果需要，可以修改为适当的标签
# plt.ylabel("FEV1_Pred")  # 如果需要，可以修改为适当的标签
# plt.title(f"r = {r:.2f}, p-value = {p_value:.2e}")
#
# plt.show()










#
# ############散点图2   FEV1--MAE
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr
#
# def average_every_n(data, n=2):
#     """接收一个 Series 并返回一个新的 Series，其中每 n 个元素的平均值"""
#     return data.groupby(data.index // n).mean()
#
# # 读取 Excel 文件
# file_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/散点图FEV1_偏差/FEV1_Pred——MAE/剔除过大的/COPD严重程度（剔除过大的）.xlsx'  # 替换为你的文件路径
# data = pd.read_excel(file_path)
#
# # 假设第一列是 y（纵轴数据），第二列是 x（横轴数据）
# y = data.iloc[:, 0]  # 第一列
# x = data.iloc[:, 1]  # 第二列
#
# # 设置 n 的值
# n = 1 # 你可以根据需要更改这个值
#
# # 对数据进行处理：每 n 个数据取平均值
# y_avg = average_every_n(y, n)
# x_avg = average_every_n(x, n)
#
# # 计算 Pearson 相关系数 r 和 P 值
# r, p_value = pearsonr(x_avg, y_avg)
#
# # 创建图表
# plt.figure(figsize=(6,6))
# sns.kdeplot(x=x_avg, y=y_avg, cmap="rainbow", fill=True)
# plt.scatter(x_avg, y_avg, color='blue', alpha=0.5, s=10)
#
# # 1:1 参考线
# plt.plot([min(x_avg), max(x_avg)], [min(x_avg), max(x_avg)], 'k--', lw=2)
#
# # 设定标题和标签
# plt.xlabel("Body Age Gap")  # 如果需要，可以修改为适当的标签
# plt.ylabel("FEV1_Pred")  # 如果需要，可以修改为适当的标签
# plt.title(f"r = {r:.2f}, p-value = {p_value:.2e}")
#
# plt.show()
#












#
# ############散点图2   FEV1--标签年龄
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr
#
# def average_every_n(data, n=2):
#     """接收一个 Series 并返回一个新的 Series，其中每 n 个元素的平均值"""
#     return data.groupby(data.index // n).mean()
#
# # 读取 Excel 文件
# file_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/test-copd真实编年相关性图/copd.xlsx'  # 替换为你的文件路径
# data = pd.read_excel(file_path)
#
# # 假设第一列是 y（纵轴数据），第二列是 x（横轴数据）
# y = data.iloc[:, 0]  # 第一列
# x = data.iloc[:, 1]  # 第二列
#
# # 设置 n 的值
# n = 20  # 你可以根据需要更改这个值
#
# # 对数据进行处理：每 n 个数据取平均值
# y_avg = average_every_n(y, n)
# x_avg = average_every_n(x, n)
#
# # 计算 Pearson 相关系数 r 和 P 值
# r, p_value = pearsonr(x_avg, y_avg)
#
# # 创建图表
# plt.figure(figsize=(6,6))
# sns.kdeplot(x=x_avg, y=y_avg, cmap="rainbow", fill=True)
# plt.scatter(x_avg, y_avg, color='blue', alpha=0.5, s=10)
#
# # 1:1 参考线
# # plt.plot([min(x_avg), max(x_avg)], [min(x_avg), max(x_avg)], 'k--', lw=2)
#
# # 设定标题和标签
# plt.xlabel("Body Age Gap")  # 如果需要，可以修改为适当的标签
# plt.ylabel("FEV1_Pred")  # 如果需要，可以修改为适当的标签
# plt.title(f"r = {r:.2f}, p-value = {p_value:.2e}")
#
# plt.show()










#########散点校正PRED F/F
# ############散点图1   FEV1--MAE
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, gaussian_kde
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
#
# from matplotlib import rcParams
# def average_every_n(data, n=2):
#     """接收一个 Series 并返回一个新的 Series，其中每 n 个元素的平均值"""
#     return data.groupby(data.index // n).mean()
# rcParams.update({
#     'figure.dpi': 500,
#
# })
# # 读取 Excel 文件
# file_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/新的Pred绘图/新新的抽取/f比f/f比f校正-预测.xlsx'  # 替换为你的文件路径
# data = pd.read_excel(file_path)
#
# # 假设第一列是 y（纵轴数据），第二列是 x（横轴数据）
# y = data.iloc[:, 0]  # 第一列
# x = data.iloc[:, 1]  # 第二列
#
# # 设置 n 的值
# n = 1  # 你可以根据需要更改这个值
#
# # 对数据进行处理：每 n 个数据取平均值
# y_avg = average_every_n(y, n)
# x_avg = average_every_n(x, n)
#
# # 计算 Pearson 相关系数 r 和 P 值
# r, p_value = pearsonr(x_avg, y_avg)
#
# # 计算样本整体的回归斜率
# slope, _ = np.polyfit(x_avg, y_avg, 1)
#
# # 计算高斯核密度估计，确定散点图中密度最高的点
# xy = np.vstack([x_avg, y_avg])
# kde = gaussian_kde(xy)
# # 构造网格
# x_grid = np.linspace(x_avg.min(), x_avg.max(), 100)
# y_grid = np.linspace(y_avg.min(), y_avg.max(), 100)
# X, Y = np.meshgrid(x_grid, y_grid)
# Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
# max_index = np.unravel_index(np.argmax(Z), Z.shape)
# densest_x = X[max_index]
# densest_y = Y[max_index]
#
#
# # 计算密度值（每个点的）
# xy_sample = np.vstack([x_avg, y_avg])
# point_densities = kde(xy_sample)
#
# # 绘制打点，点颜色表示密度（渐变色效果）
# plt.figure(figsize=(13.5,12))
# ax = plt.gca()
# color_values = x_grid + y_grid  # 可以根据需要更改此计算函数
#
# # 使用"coolwarm" colormap（从红色到蓝色）来实现中心红色，外部蓝色的渐变
# cmap = plt.get_cmap("coolwarm")
# norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
# sm = ScalarMappable(cmap=cmap, norm=norm)
#
# sc = plt.scatter(x_avg, y_avg, c=point_densities, cmap="coolwarm", s=100, alpha=0.8)
#
# # 添加颜色条
# # plt.colorbar(sc, label='Density')
#
# # 获取坐标轴范围
# xlims = ax.get_xlim()
#
# # 绘制密度最高点为中心的回归线
# x_line = np.array(xlims)
# y_line = slope * (x_line - densest_x) + densest_y
# plt.plot(x_line, y_line, color='black', lw=4, linestyle='-')
#
#
# # 设定标题和标签
# # 设定标题和标签（y 控制标题垂直位置）
#
# plt.xlabel("Predicted Age", fontsize=45)
# #Chronological Age
# #Lung age gap
# #
# plt.ylabel("Calibrated FEV1/FVC_x", fontsize=45)
# #Calibrated FEV1/FEV1_Pred
# #FEV1/FEV1_Pred
# # FEV1/FVC_x
# #Calibrated FEV1
# #Calibrated FVC
# #
# #Chronological Age
#
#
# plt.tick_params(axis='x', labelsize=45)
# plt.tick_params(axis='y', labelsize=45)
#
# plt.title(f"r = {r:.2f}, p-value = {p_value:.2e}", y=0.90, fontsize = 45)  # 将标题位置向下移动
# # plt.ylim(1,120)
# # plt.xlim(30,80)
# ######Lung age gap
# # plt.ylim(0.21,0.8)
# # plt.ylim(1,120)
# # # plt.xlim(-15,15)
# # plt.xlim(30,80)
# # plt.xlim(30,80)
# #######FVC
# # plt.ylim(0.8,5.5)
# # plt.xlim(30,80)
# # plt.xlim(-15,15)
# #####FEV1
# # plt.ylim(0.1,3.8)
# # # plt.xlim(30,80)
# # plt.xlim(-15,15)
# ######f/f
# plt.ylim(0.21,0.8)
# # plt.xlim(30,80)
# # 手动设置 x 轴刻度（例如每 10 个单位显示一个刻度）
# # plt.xticks(np.arange(-15, 16, 5))  # -15  -10  -5  0  5  10  15
# plt.xticks(np.arange(30, 81, 10))  # 30, 40, 50, 60, 70, 80
# # plt.yticks(np.arange(0.1, 3.8, 6))  # 30, 40, 50, 60, 70, 80
# # plt.yticks([1,1.9,2.8,3.7,4.6,5.5])
# plt.show()










# # # # ######颜色条
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.colors as colors
# from matplotlib import rcParams
#
# rcParams.update({
#     'figure.dpi': 100,
# })
#
# # 设置密度范围，根据你的 KDE 输出设置 vmin/vmax
# norm = colors.Normalize(vmin=0, vmax=0.05)
# cmap = cm.get_cmap("coolwarm")
# mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
# mappable.set_array([])  # 必须加这句，否则也会报错
#
# # 创建图和子图（子图是我们显式告诉 colorbar 的 ax）
# fig, ax = plt.subplots(figsize=(2.6, 13.5))
#
# # 添加颜色条到指定坐标轴
# cb = plt.colorbar(mappable, cax=ax, orientation='vertical')
#
# cb.set_label('Density', fontsize=45)
# cb.ax.tick_params(labelsize=40)
#
# plt.tight_layout()  # 自动调整边距
# # 隐藏坐标轴刻度线
# # ax.tick_params(size=0)
# output_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/test-copd真实编年相关性图/color_bar2.png'
# plt.savefig(output_path, bbox_inches='tight', dpi=600)
# # plt.show()








# ######之前的色块相关性图
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr, gaussian_kde
# from matplotlib.path import Path
# from matplotlib import rcParams
# rcParams.update({
#     'figure.dpi': 600,
# })
# def average_every_n(data, n=2):
#     """接收一个 Series 并返回一个新的 Series，其中每 n 个元素的平均值"""
#     return data.groupby(data.index // n).mean()
#
# # 读取 Excel 文件
# file_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/新的Pred绘图/新新的抽取/Pred/Pred校正-Lung age gap.xlsx'  # 替换为你的文件路径
# data = pd.read_excel(file_path)
#
# # 假设第一列是 y（纵轴数据），第二列是 x（横轴数据）
# y = data.iloc[:, 0]  # 第一列
# x = data.iloc[:, 1]  # 第二列
#
# # 设置 n 的值
# n = 1  # 你可以根据需要更改这个值
#
# # 对数据进行处理：每 n 个数据取平均值
# y_avg = average_every_n(y, n)
# x_avg = average_every_n(x, n)
#
# # 计算 Pearson 相关系数 r 和 P 值
# r, p_value = pearsonr(x_avg, y_avg)
#
# # 计算样本整体的回归斜率
# slope, _ = np.polyfit(x_avg, y_avg, 1)
#
# # 计算高斯核密度估计，确定散点图中密度最高的点
# xy = np.vstack([x_avg, y_avg])
# kde = gaussian_kde(xy)
# # 构造网格
# x_grid = np.linspace(x_avg.min(), x_avg.max(), 100)
# y_grid = np.linspace(y_avg.min(), y_avg.max(), 100)
# X, Y = np.meshgrid(x_grid, y_grid)
# Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
# max_index = np.unravel_index(np.argmax(Z), Z.shape)
# densest_x = X[max_index]
# densest_y = Y[max_index]
#
# # 创建图表并绘制密度图与散点图
# plt.figure(figsize=(10,10))
# ax = plt.gca()
#
# # # 绘制带填充的 KDE 密度图，并保存返回的 contour set
# # kde_plot = sns.kdeplot(x=x_avg, y=y_avg, cmap="rainbow", fill=True)
# #
# # # 添加颜色条
# # mappable = kde_plot.collections[0]
# # plt.colorbar(mappable, ax=ax, label='Density')
#
# # 绘制带填充的 KDE 密度图
# sns.kdeplot(x=x_avg, y=y_avg, cmap="rainbow", fill=True)
# #plt.scatter(x_avg, y_avg, c=y_avg, cmap='', alpha=0.5, s=10)#渐变色
# plt.scatter(x_avg, y_avg, color='darkgray', alpha=0.6, s=30)
# #darkgray
#
#
# # 获取当前坐标轴的 x 轴范围
# xlims = ax.get_xlim()
#
# # 计算通过密度最高点且斜率为样本整体斜率的直线端点
# x_line = np.array(xlims)
# y_line = slope * (x_line - densest_x) + densest_y
#
# # 绘制直线，并设置较高的 zorder 使其位于最上层
# line, = plt.plot(x_line, y_line, color='black', lw=4, linestyle='-', zorder=10)
#
# # 合并 KDE 填充区域所有的 Path 为 compound path
# filled_paths = ax.collections[0].get_paths()
# if filled_paths:
#     compound_path = Path.make_compound_path(*filled_paths)
#     line.set_clip_path(compound_path, transform=ax.transData)
#
# plt.tick_params(axis='x', labelsize=30)
# plt.tick_params(axis='y', labelsize=30)
#
# # 设定标题和标签
# # 设定标题和标签（y 控制标题垂直位置）
# plt.xlabel("Lung age gap",fontsize = 30)
# #Chronological Age
# #
# #Predicted Age
# plt.ylabel("Calibrated FEV1/FEV1_Pred",fontsize = 30)
# #Calibrated FEV1/FVC_x
# #FEV1/FEV1_Pred
# # FEV1/FVC_x
# #Calibrated FEV1
# #Calibrated FVC
# #
# #
#
#
# plt.title(f"r = {r:.2f}, p-value = {p_value:.2e}", y=0.94,fontsize = 34)  # 将标题位置向下移动
# # plt.ylim(1,125)
# # plt.xlim(30,80)
# ####预测
# plt.ylim(1,125)
# # plt.xlim(25,85)
# plt.xlim(-15,15)
# ####f/f
# # plt.ylim(0.21,0.85)
# # plt.xlim(25,85)
# # plt.xlim(-15,15)
# ######fvc
# # plt.ylim(0.8,5.2)
# # # plt.xlim(25,85)
# # plt.xlim(-15,15)
# #####fev1
# # plt.ylim(0.1,3.8)
# # plt.xlim(25,85)
# # plt.xlim(-15,15)
# plt.show()










# ###只绘制颜色条
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.colors as colors
# from matplotlib import rcParams
#
# rcParams.update({
#     'figure.dpi': 600,
# })
# # 设置密度范围，根据你的 KDE 输出设置 vmin/vmax
# norm = colors.Normalize(vmin=0, vmax=0.05)
# cmap = cm.get_cmap("rainbow")
# mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
# mappable.set_array([])  # 必须加这句，否则也会报错
#
# # 创建图和子图（子图是我们显式告诉 colorbar 的 ax）
# fig, ax = plt.subplots(figsize=(1.5, 8))
#
# # 添加颜色条到指定坐标轴
# cb = plt.colorbar(mappable, cax=ax, orientation='vertical')
#
# cb.set_label('Density', fontsize=16)
# cb.ax.tick_params(labelsize=16)
#
# plt.tight_layout()  # 自动调整边距
# # 隐藏坐标轴刻度线
# # ax.tick_params(size=0)
# plt.show()








# import itk
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 读取 JPG 格式的 CT 图像（使用 ITK）
# image = itk.imread("/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/新的Pred绘图/新新的抽取/fev1/13/5.png", itk.F)  # 读取为 float 类型
#
# # 将 ITK 图像转换为 NumPy 数组
# array = itk.GetArrayFromImage(image)
#
# # 归一化到 0-255
# array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
# array = array.astype(np.uint8)
#
# # 应用伪彩色映射（Matplotlib 的 'jet' 映射）
# plt.imshow(array, cmap='jet')  # 可改成 'hot', 'coolwarm', 'magma'
# plt.axis('off')  # 关闭坐标轴
# plt.show()














#
# # ########bland-altman绘图
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# from pingouin import intraclass_corr
# from matplotlib.lines import Line2D
#
# # 设置全局字体
# plt.rcParams.update({
#     'figure.dpi': 100,
#     'font.size': 45,
#     'axes.titlesize': 45,
#     'axes.labelsize': 45,
#     'xtick.labelsize': 45,
#     'ytick.labelsize': 45,
#     'legend.fontsize': 45
# })
#
# # 读取Excel文件
# file_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/test-copd真实编年相关性图/test.xlsx'
# df = pd.read_excel(file_path)
#
# data1 = df.iloc[:, 0]
# data2 = df.iloc[:, 1]
#
# # Bland-Altman 分析
# mean = (data1 + data2) / 2
# diff = data1 - data2
# mean_diff = np.mean(diff)
# std_diff = np.std(diff, ddof=1)
# upper_limit = mean_diff + 1.96 * std_diff
# lower_limit = mean_diff - 1.96 * std_diff
#
# # 计算量化指标
# rmse = np.sqrt(mean_squared_error(data1, data2))
# bias = mean_diff
# cv = std_diff / np.mean(mean)
# # 将数据转换成长格式
# data_long = pd.melt(df.reset_index(), id_vars=['index'], value_vars=[df.columns[0], df.columns[1]],
#                     var_name='Rater', value_name='Score')
#
# # 计算 ICC（注意调整参数）
# icc = intraclass_corr(data=data_long, targets='index', raters='Rater', ratings='Score').iloc[0]['ICC']
# print(f"Intraclass Correlation Coefficient (ICC): {icc:.4f}")
#
# loa_width = upper_limit - lower_limit
#
#
# plt.figure(figsize=(20, 16))
# plt.scatter(mean, diff, c='#EA9010', alpha=0.8, s=200)
# ##EA9010
#
# # 画三条线
# plt.axhline(mean_diff, color='blue', linestyle='-', linewidth=3)
# plt.axhline(upper_limit, color='brown', linestyle='--', linewidth=3)
# plt.axhline(lower_limit, color='brown', linestyle='--', linewidth=3.5)
#
# # 添加文本标注（线条上下分别标注描述和数值）
# plt.text(75, upper_limit + 0.7, '+1.96 SD', fontsize=45, color='black')
# plt.text(80, upper_limit - 2.1, f'{upper_limit:.1f}', fontsize=45, color='black')
#
# plt.text(79, mean_diff + 0.7, 'Mean', fontsize=45, color='black')
# plt.text(80.9, mean_diff - 2.1, f'{mean_diff:.1f}', fontsize=45, color='black')
#
# plt.text(76, lower_limit + 0.7, f'-1.96 SD', fontsize=45, color='black')
# plt.text(80.6, lower_limit - 2.1, f'{lower_limit:.1f}', fontsize=45, color='black')
#
# # 设置顶部指标文字
# plt.text(82, upper_limit + 9, f'ICC= {icc:.4f}', fontsize=45, ha='right')
# plt.text(66, upper_limit + 9, f'RMSE= {rmse:.4f}', fontsize=45, ha='right')
# plt.text(48, upper_limit + 9, f'CV= {cv:.2%}', fontsize=45, ha='right')
#
# # 坐标轴设置
# plt.ylim(-20, 20)
# plt.xlim(33,85)
# plt.xlabel('Means')
# plt.ylabel('Differences')
# plt.title('')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# # output_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/第一步/resnet34/resnet_34/test.png'
# # plt.savefig(output_path, bbox_inches='tight', dpi=600)








# ##########复制过来的PRED
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, gaussian_kde
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
#
# from matplotlib import rcParams
#
#
# # def format_sci_notation(value):
# #     """将科学计数法格式转换为×10格式，指数显示在右上角"""
# #     # 检查是否是NaN
# #     if np.isnan(value):
# #         return "NaN"
# #
# #     # 处理负值（特别是相关系数r）
# #     if value < 0:
# #         # 对于负值，使用常规格式显示
# #         return f"{value:.2f}"
# #
# #     # 处理非负值（特别是p-value）
# #     if value < 0.001:
# #         exponent = int(np.floor(np.log10(value)))
# #         coefficient = value / (10 ** exponent)
# #         # 使用 LaTeX 格式使指数显示为右上角
# #         return r"${:.2f} \times 10^{{{}}}$".format(coefficient, exponent)
# #     elif value < 0.01:
# #         return f"{value:.3f}"
# #     else:
# #         return f"{value:.2f}"
# def format_sci_notation(value):
#     """将所有值格式化为科学计数法（x.xx×10^n）"""
#     # 检查是否是NaN
#     if np.isnan(value):
#         return "NaN"
#
#     # 0的特殊处理
#     if value == 0:
#         return "0"
#
#     # 处理所有非零值
#     exponent = int(np.floor(np.log10(abs(value))))
#     coefficient = value / (10 ** exponent)
#
#     # 使用 LaTeX 格式使指数显示为右上角
#     return r"${:.2f} \times 10^{{{}}}$".format(coefficient, exponent)
# def format_sci_notation_r(value):
#     """将所有值格式化为科学计数法（x.xx×10^n）"""
#     # 检查是否是NaN
#     if np.isnan(value):
#         return "NaN"
#
#     # 0的特殊处理
#     if value == 0:
#         return "0"
#
#     # 处理所有非零值
#     exponent = int(np.floor(np.log10(abs(value))))
#     coefficient = value / (10 ** exponent)
#
#     # 使用 LaTeX 格式使指数显示为右上角
#     return f"{coefficient:.2f}"
#
# def average_every_n(data, n=2):
#     """接收一个 Series 并返回一个新的 Series，其中每 n 个元素的平均值"""
#     return data.groupby(data.index // n).mean()
# rcParams.update({
#     'figure.dpi': 100,
#
# })
# # 读取 Excel 文件
# file_path = '/home/zsq/train/pre_process/DATE/外部验证excel/其余关系/三个外部合一起/pred-编年.xlsx'  # 替换为你的文件路径
# data = pd.read_excel(file_path)
#
# # 假设第一列是 y（纵轴数据），第二列是 x（横轴数据）
# y = data.iloc[:, 0]  # 第一列
# x = data.iloc[:, 1]  # 第二列
#
# # 设置 n 的值
# n = 1  # 你可以根据需要更改这个值
#
# # 对数据进行处理：每 n 个数据取平均值
# y_avg = average_every_n(y, n)
# x_avg = average_every_n(x, n)
#
# # 计算 Pearson 相关系数 r 和 P 值
# r, p_value = pearsonr(x_avg, y_avg)
#
# # 计算样本整体的回归斜率
# slope, _ = np.polyfit(x_avg, y_avg, 1)
#
# # 计算高斯核密度估计，确定散点图中密度最高的点
# xy = np.vstack([x_avg, y_avg])
# kde = gaussian_kde(xy)
# # 构造网格
# x_grid = np.linspace(x_avg.min(), x_avg.max(), 100)
# y_grid = np.linspace(y_avg.min(), y_avg.max(), 100)
# X, Y = np.meshgrid(x_grid, y_grid)
# Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
# max_index = np.unravel_index(np.argmax(Z), Z.shape)
# densest_x = X[max_index]
# densest_y = Y[max_index]
#
#
# # 计算密度值（每个点的）
# xy_sample = np.vstack([x_avg, y_avg])
# point_densities = kde(xy_sample)
#
# # 绘制打点，点颜色表示密度（渐变色效果）
# # plt.figure(figsize=(13.5,12))
# # plt.figure(figsize=(16.2,16))#######pred的
# plt.figure(figsize=(16,16))#######其余的
# ax = plt.gca()
# color_values = x_grid + y_grid  # 可以根据需要更改此计算函数
#
# # 使用"coolwarm" colormap（从红色到蓝色）来实现中心红色，外部蓝色的渐变
# cmap = plt.get_cmap("coolwarm")
# norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
# sm = ScalarMappable(cmap=cmap, norm=norm)
#
# sc = plt.scatter(x_avg, y_avg, c=point_densities, cmap="coolwarm", s=200, alpha=0.8)
#
# # 添加颜色条
# # plt.colorbar(sc, label='Density')
#
# # 获取坐标轴范围
# xlims = ax.get_xlim()
#
# # 绘制密度最高点为中心的回归线
# x_line = np.array(xlims)
# y_line = slope * (x_line - densest_x) + densest_y
# plt.plot(x_line, y_line, color='black', lw=4, linestyle='-')
#
#
# # 设定标题和标签
# # 设定标题和标签（y 控制标题垂直位置）
#
# plt.xlabel("Chronological age", fontsize=45)
# #Calibrated lung age gap
# #
# #Bio-LungAge
# plt.ylabel("Calibrated FEV$_{1}$%pred", fontsize=45)
# #
# #Calibrated FEV$_{1}$/FVC
# #Calibrated FEV$_{1}$
# #Calibrated FVC
# #
#
#
# plt.tick_params(axis='x', labelsize=45)
# plt.tick_params(axis='y', labelsize=45)
#
# # plt.title(f"r = {r:.2f}, p-value = {p_value:.2e}", y=0.90, fontsize = 45)  # 将标题位置向下移动
# # 修改这部分代码
# plt.title(f"r = {r:.2f}, "
#           f"p-value = {format_sci_notation(p_value)}",
#           y=0.90, fontsize=45)
#
#
# plt.ylim(1,120)
# # plt.ylim(1,100)
# # plt.xlim(30,80)
# # plt.ylim(31,89)
# plt.xlim(30,90)
# # plt.ylim(0.11,0.8)
# ######Lung age gap
# # plt.ylim(0.21,0.8)
# # plt.ylim(1,120)
# # plt.xlim(-20,20)
# # plt.ylim(30, 88)  # y 轴范围从 0 到 6（确保 0-1.5 的空间存在）
# # plt.yticks(np.arange(40, 81, 10))  #
# #
# # plt.xlim(30,80)
# # plt.xticks(np.arange(30, 81, 10))
# # ######FVC
# # plt.ylim(0.8,5.5)
# # plt.xlim(30,90)
# # plt.xlim(-15,15)
# # ####FEV1
# # plt.ylim(0.1,3.8)
# # plt.ylim(0.1,6)
# # plt.xlim(30,80)
# # plt.xlim(-25,25)
# # #####f/f
# # plt.ylim(0.21,0.9)
# # plt.ylim(0.21,0.8)
# # plt.ylim(0.61,1.1)
# # plt.xlim(30,80)
# # ################手动设置 x 轴刻度（例如每 10 个单位显示一个刻度）
# # plt.xticks(np.arange(-15, 16, 5))  # -15  -10  -5  0  5  10  15
# # plt.xticks(np.arange(-30, 31, 10))  # -15  -10  -5  0  5  10  15
# # plt.xticks(np.arange(-25, 26, 10))  #
# # plt.xticks(np.arange(-20, 21, 5))  #
# # plt.xticks(np.arange(30, 81, 10))  # 30, 40, 50, 60, 70, 80
#
# # plt.ylim(0.2,3.8)
# # plt.yticks(np.arange(0.8, 3.81, 0.6))  # 30, 40, 50, 60, 70, 80
#
# # plt.ylim(0.08,0.8)
# # plt.yticks(np.arange(0.2, 0.81, 0.12))  # 30, 40, 50, 60, 70, 80
#
# # plt.ylim(0.1,5.5)
# # plt.yticks(np.arange(1.0, 5.51, 0.9))  # 30, 40, 50, 60, 70, 80
#
# # plt.ylim(0,4.5)
# # plt.yticks(np.arange(0.9, 4.51, 0.9))  # 30, 40, 50, 60, 70, 80
#
# # plt.ylim(0.68,1.1)
# # plt.yticks(np.arange(0.68, 1.11, 0.07))  # 30, 40, 50, 60, 70, 80
#
# # plt.ylim(0.2, 3.2)  # y 轴范围从 0 到 6（确保 0-1.5 的空间存在）
# # plt.yticks(np.arange(0.8, 3.21, 0.6))  #
#
# # plt.ylim(0.4, 5.5)  # y 轴范围从 0 到 6（确保 0-1.5 的空间存在）
# # plt.yticks(np.arange(1.5, 5.6, 1))  #
# #
# # plt.yticks([1,1.9,2.8,3.7,4.6,5.5])
# ################## 保存图片到指定路径
# output_path = '/home/zsq/train/pre_process/DATE/外部验证excel/其余关系/三个外部合一起/合一起2/pred-编年.png'
# plt.savefig(output_path, bbox_inches='tight', dpi=600)
#
# # plt.show()










# #######随机抽取G1和G2
# import pandas as pd
#
# # 读取Excel文件
# file_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/COPD/COPD严重程度/G2.xlsx'
# df = pd.read_excel(file_path)
#
# # 定义抽样数量
# n = 41
#
# # 随机抽取 n 行数据（不放回）
# sampled_df = df.sample(n=n, random_state=42)  # 设置 random_state 保证每次抽样结果一致
#
# # 输出抽样结果
# print(sampled_df)
#
# # 可选择将抽样结果保存到新的Excel文件
# sampled_df.to_excel('/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/新的Pred绘图/G2抽样.xlsx', index=False)









#######处理txt，只取MAE的值
#
# # 定义输入和输出文件的路径
# input_file_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_101_40pin/resnet_101/output_error60.txt'
# output_file_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_101_40pin/resnet_101/output.txt'
#
# # 打开输入文件进行读取，打开输出文件进行写入
# with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
#     # 逐行读取输入文件
#     for line in infile:
#         # 查找'Current Mean Error: '字符串的位置
#         index = line.find('Current Mean Error: ')
#         if index != -1:
#             # 如果找到了，保留该字符串后面的所有内容，并去掉末尾的换行符
#             result = line[index + len('Current Mean Error: '):].strip()
#             # 将结果写入输出文件，不添加额外的换行符（因为line本身可能包含一个换行符）
#             outfile.write(result)
#             # 在每行处理后手动添加一个换行符以分隔不同的记录
#             outfile.write('\n')













# #######多个样本
# #######计算相似度的p和F
# import numpy as np
# from scipy.stats import f
#
# # 数据
# means = [97.05, 97.20, 87.65, 68.91, 40.22, 74] # 各组均值
# stds = [9.15, 9.95, 5.43, 7.75, 7.25, 14.30] # 各组标准差
# ns = [2571, 656, 208, 292, 41, 541] # 各组样本数
#
# # 计算组间平方和 (SSB) 和组内平方和 (SSW)
# grand_mean = np.average(means, weights=ns) # 总体平均值
# ssb = sum([n * (m - grand_mean)**2 for n, m in zip(ns, means)]) # 组间平方和
# total_n = sum(ns) # 总样本数
# ssw = sum([(n-1)*s**2 for n, s in zip(ns, stds)]) # 组内平方和
#
# # 自由度
# df_between = len(means) - 1 # 组间自由度
# df_within = total_n - len(means) # 组内自由度
#
# # 均方
# msb = ssb / df_between # 组间均方
# msw = ssw / df_within # 组内均方
#
# # F值
# f_value = msb / msw
#
# # p值
# p_value = 1 - f.cdf(f_value, df_between, df_within)
#
# print(f"F值: {f_value}")
# print(f"p值: {p_value}")










#####用服务器画高分辨率图可能会陷入“假卡死”
######bar图
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from matplotlib import rcParams
#
# # 全局字体设置（确保所有元素都变大）
# rcParams.update({
#     'figure.dpi': 100,
#     'font.size': 24,           # 增大基准字体大小
#     'axes.titlesize': 24,      # 标题
#     'axes.labelsize': 24,      # 坐标轴标签（x轴和y轴）
#     'xtick.labelsize': 24,     # X轴刻度
#     'ytick.labelsize': 24,     # Y轴刻度
#     'legend.fontsize': 24,     # 图例
#     'legend.title_fontsize': 24 # 图例标题
# })
#
# # 数据
# data = {
#     "Term": [
#         "ResNet10", "ResNet18", "ResNet34", "ResNet50", "ResNet101",
#         "20", "40", "60",
#         "30-78", "35-74"
#     ],
#     "MAE": [
#         4.01, 4.13, 3.96, 4.46, 4.85,
#         4.12,3.96 ,3.71 ,
#         3.71, 3.54
#     ],
#     "Type": [
#         "Network type", "Network type", "Network type", "Network type", "Network type",
#          "Slice count", "Slice count","Slice count",
#         "Age range", "Age range"
#     ]
# }
#
# df = pd.DataFrame(data)
#
# # 设定绘图风格
# sns.set(style="whitegrid")
#
# # 创建条形图（增大图形尺寸以适应大字体）
# plt.figure(figsize=(16, 10))  # 关键修正：在这里启用 constrained_layout
#
# ax = sns.barplot(data=df, y="Term", x="MAE", hue="Type", dodge=False,
#                 palette={"Network type": "#E74C3C","Slice count": "#5DADE2","Age range": "#1ABC9C"})###   #EA9010
# from tqdm import tqdm
# # 在每个柱子上添加 MAE 的值（增大字体）
# for p in tqdm(ax.patches):
#     width = p.get_width()
#     y = p.get_y() + p.get_height() / 2
#     ax.text(width + 0.04, y, f'{width:.2f}',
#             va='center', ha='left',
#             fontsize=24,  # 增大数值标签字体
#             bbox=dict(facecolor='white', alpha=0.7, pad=0.3))  # 增大背景框
#
# # 设置标签和标题（显式覆盖设置）
# ax.set_xlabel("MAE", fontsize=24)  # 进一步增大x轴标签
# ax.set_ylabel("", fontsize=24)     # y轴标签
#
# # 设置图例（显式覆盖设置）
# legend = ax.legend(title="", fontsize=22)  # 增大图例字体
# for text in legend.get_texts():
#     text.set_fontsize(22)  # 确保所有图例文本一致
#
# # 设置刻度标签（显式覆盖设置）
# ax.tick_params(axis='x', labelsize=24)  # 增大x轴刻度
# ax.tick_params(axis='y', labelsize=24)  # 增大y轴刻度
#
# # 调整 x 轴范围
# plt.xlim(left=min(df["MAE"]) - 0.5)
#
# # 调整布局防止文字被截断
# plt.tight_layout()
#
# # 显示图形
# # plt.show()
# output_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/第一步/resnet50/tu.png'
# plt.savefig(output_path,  dpi=600)
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from matplotlib import rcParams
#
# # 全局字体设置
# rcParams.update({
#     'figure.dpi': 100,
#     'font.size': 24,
#     'axes.titlesize': 24,
#     'axes.labelsize': 24,
#     'xtick.labelsize': 24,
#     'ytick.labelsize': 24,
#     'legend.fontsize': 24,
#     'legend.title_fontsize': 24
# })
#
# # 数据
# data = {
#     "Term": [
#         "ResNet10", "ResNet18", "ResNet34", "ResNet50", "ResNet101",
#         "20", "40", "60",
#         "30-78", "35-74"
#     ],
#     "MAE": [
#         4.01, 4.13, 3.96, 4.46, 4.75,
#         4.12, 3.96, 3.71,
#         3.71, 3.54
#     ],
#     "Type": [
#         "Network type", "Network type", "Network type", "Network type", "Network type",
#         "Slice count", "Slice count", "Slice count",
#         "Age range", "Age range"
#     ]
# }
#
# df = pd.DataFrame(data)
#
# # 设定绘图风格
# sns.set(style="whitegrid")
#
# # 创建条形图
# plt.figure(figsize=(16, 10))
# ax = sns.barplot(data=df, y="Term", x="MAE", hue="Type", dodge=False,
#                 palette={"Network type": "#E74C3C", "Slice count": "#5DADE2", "Age range": "#1ABC9C"})
#
# # 在每个柱子上添加 MAE 的值
# for p in ax.patches:
#     width = p.get_width()
#     y = p.get_y() + p.get_height() / 2
#     ax.text(width + 0.04, y, f'{width:.2f}',
#             va='center', ha='left',
#             fontsize=24,
#             bbox=dict(facecolor='white', alpha=0.7, pad=0.3))
#
# # 设置标签和标题
# ax.set_xlabel("MAE", fontsize=24, labelpad=10)  # 添加labelpad参数调整位置
# ax.set_ylabel("", fontsize=24)
#
# # 设置图例
# legend = ax.legend(title="", fontsize=22)
# for text in legend.get_texts():
#     text.set_fontsize(22)
#
# # 设置刻度标签
# ax.tick_params(axis='x', labelsize=24)
# ax.tick_params(axis='y', labelsize=24)
#
# # 调整 x 轴范围
# plt.xlim(left=min(df["MAE"]) - 0.5)
#
# # 关键修改：将x轴标签向左移动
# # 方法1：使用set_label_coords调整位置（推荐）
# ax.xaxis.set_label_coords(0.435, -0.06)  # 将x轴标签向左移动（第一个参数越小越向左）
# plt.xlim(3,5)
# # 方法2：使用文本标签替代（备选方案）
# # ax.set_xlabel("")  # 先移除默认标签
# # ax.text(0.0, -0.1, "MAE", transform=ax.transAxes, fontsize=24)  # 在左下角添加自定义标签
#
# # 调整布局防止文字被截断
# plt.tight_layout()
#
# # 保存图形
# output_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/第一步/resnet50/tu.png'
# plt.savefig(output_path, dpi=600)  # 添加bbox_inches='tight'确保所有元素可见










# # # ###########新的pred差异
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置全局字体大小（基准值，可被局部覆盖）
# rcParams.update({
#     'figure.dpi': 100,  # 降低 DPI 以避免显示问题
#     'font.size': 16,           # 默认字体大小（影响大多数文本）
#     'axes.titlesize': 18,      # 标题字体大小
#     'axes.labelsize': 18,      # 坐标轴标签大小
#     'xtick.labelsize': 18,      # X轴刻度标签大小
#     'ytick.labelsize': 18,      # Y轴刻度标签大小
#     'legend.fontsize': 16     # 图例字体大小
# })
# # 示例数据
# groups = ['35-43', '44-50', '51-53', '54-56', '57-60', '61-65', '66-74']  # 组别
# means_left = [79.34, 77.43, 79.07, 75.91, 76.64, 77.14, 71.66]  # 左柱均值
# stds_left = [8.74, 9.9, 9.43, 15.1, 15.36, 13.16, 18.11]  # 左柱标准差
# means_right = [73.40, 72.06, 73.22, 72.2, 70.17, 69.06, 64.47]  # 右柱均值
# stds_right = [15.44, 14.27, 12.48, 12.1, 16.87, 17.08, 14.54]  # 右柱标准差
#
# # 可调整参数
# bar_width = 0.6  # 单根柱子的宽度
# group_spacing = 1.7  # 组间距离 (1.0是默认值，增大此值会增加组间距离)
#
# x = np.arange(len(groups)) * group_spacing  # 组的x位置，乘以组间距离系数
# width = bar_width  # 柱子的宽度
#
# fig, ax = plt.subplots(figsize=(12, 7))  # 进一步增加图形大小
#
#
# ########
# cmap = plt.get_cmap("coolwarm")
# # color1 = cmap(0.15)    # 最冷色（深蓝）
# # color2 = cmap(0.85)    # 最暖色（深红）
# color1 = '#AAC4FE'    #
# color2 = '#F18E70'   #
# # 绘制柱状图
# left_bars = ax.bar(x - width/2, means_left, width, yerr=stds_left, label='Lower', capsize=5, color = color1)
# right_bars = ax.bar(x + width/2, means_right, width, yerr=stds_right, label='Higher', capsize=5, color = color2)
#
# # 添加文本标签
# # def add_labels(bars, stds):
# #     for bar, std in zip(bars, stds):
# #         height = bar.get_height()
# #         ax.text(bar.get_x() + bar.get_width()/2., height + 1,
# #                 f'{height:.1f}±{std:.1f}',
# #                 ha='center', va='bottom', fontsize=15)
# #
# # add_labels(left_bars, stds_left)
# # add_labels(right_bars, stds_right)
#
# # 添加文本标签
# def add_labels_left(bars, stds):
#     for idx,(bar, std) in enumerate(zip(bars, stds)):
#         if(idx == 3 or idx == 6):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.08 + bar.get_width()/2., height+2 ,
#                 f'{height:.1f}±{std:.1f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif(idx==0 or idx == 1 or idx == 2):
#             height = bar.get_height()
#             ax.text(bar.get_x() - 0.15 + bar.get_width() / 2., height,
#                     f'{height:.1f}±{std:.1f}',
#                     ha='center', va='bottom', fontsize=14)
#         else:
#             height = bar.get_height()
#             ax.text(bar.get_x() - 0.08 + bar.get_width() / 2., height,
#                     f'{height:.1f}±{std:.1f}',
#                     ha='center', va='bottom', fontsize=14)
#
#
# def add_labels_right(bars, stds):
#     for bar, std in zip(bars, stds):
#         height = bar.get_height()
#         ax.text(bar.get_x() + 0.23 + bar.get_width()/2., height ,
#                 f'{height:.1f}±{std:.1f}',
#                 ha='center', va='bottom', fontsize=14)
#
#
# # def add_labels_left(bars, stds):
# #     for bar, std in zip(bars, stds):
# #         height = bar.get_height()
# #         ax.text(
# #             bar.get_x() - 0.08 + bar.get_width() / 2.,  # x 坐标
# #             height + 1,  # y 坐标
# #             f'{height:.1f}±{std:.1f}',  # 文本内容
# #             ha='center',  # 水平对齐
# #             va='bottom',  # 垂直对齐
# #             fontsize=11,  # 字体大小
# #             bbox=dict(  # 背景框设置
# #                 facecolor='white',  # 背景填充白色
# #                 edgecolor='none',  # 无边框
# #                 alpha=1,  # 透明度（可选）
# #                 boxstyle='round,pad=0.01'  # 圆角+内边距
# #             )
# #         )
# #
# # def add_labels_right(bars, stds):
# #     for bar, std in zip(bars, stds):
# #         height = bar.get_height()
# #         ax.text(
# #             bar.get_x() + 0.18 + bar.get_width() / 2.,  # x 坐标
# #             height + 1,  # y 坐标
# #             f'{height:.1f}±{std:.1f}',  # 文本内容
# #             ha='center',  # 水平对齐
# #             va='bottom',  # 垂直对齐
# #             fontsize=11,  # 字体大小
# #             bbox=dict(  # 背景框设置
# #                 facecolor='white',  # 背景填充白色
# #                 edgecolor='none',  # 无边框
# #                 alpha=1,  # 透明度（可选）
# #                 boxstyle='round,pad=0.01'  # 圆角+内边距
# #             )
# #         )
# # add_labels_left(left_bars, stds_left)
# # add_labels_right(right_bars, stds_right)
#
# # 添加组间比较的完整括号和星号
# def add_significance_brackets(ax, x1, x2, y, text, line_height=3):
#     # 绘制完整的括号形状（水平线+两端竖线）
#     bracket_height = line_height
#     # 水平线
#     ax.plot([x1, x2], [y, y], lw=1, color='k')
#     # 左端竖线
#     ax.plot([x1, x1], [y, y-bracket_height], lw=1, color='k')
#     # 右端竖线
#     ax.plot([x2, x2], [y, y-bracket_height], lw=1, color='k')
#     # 添加星号文本（位置在水平线上方）
#     ax.text((x1+x2)*0.5, y + bracket_height*0.3, text,
#             ha='center', va='bottom', fontsize=16)
#
# # 假设的显著性结果 (这里需要替换为你的实际统计结果)
# # 格式: (group_index, y_position, significance_text)
# significance_data = [
#     (0, 100, '*'),    # 第1组显著
#     (1, 100, '*'),    # 第2组显著
#     (2, 100, '*'),    # 第3组显著
#     (3, 100, '*'),    # 第4组显著
#     (4, 100, ''),     # 第5组不显著
#     (5, 100, '*'),    # 第6组显著
#     (6, 100, ''),     # 第7组不显著
# ]
#
# for i, y, text in significance_data:
#     if((i!=4)&(i!=6)):
#         x1 = x[i] - width/2 - 0.1
#         x2 = x[i] + width/2 + 0.1
#         add_significance_brackets(ax, x1, x2, y, text)
#
# add_labels_left(left_bars, stds_left)
# add_labels_right(right_bars, stds_right)
#
# ax.set_ylim(0, 110)  # 调整y轴范围以容纳显著性标记
# ax.set_xticks(x)
# ax.set_xticklabels(groups)
# ax.set_ylabel('Calibrated FEV$_{1}$%pred')
# ax.set_xlabel('Age group')
# ax.set_title('')
# ax.legend()
#
# plt.tight_layout()
# # plt.show()
# output_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/test-copd真实编年相关性图/pred1.png'
# plt.savefig(output_path, bbox_inches='tight', dpi=600)








#
# # ###########新的pred差异--external1
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置全局字体大小（基准值，可被局部覆盖）
# rcParams.update({
#     'figure.dpi': 100,  # 降低 DPI 以避免显示问题
#     'font.size': 20,           # 默认字体大小（影响大多数文本）
#     'axes.titlesize': 20,      # 标题字体大小
#     'axes.labelsize': 20,      # 坐标轴标签大小
#     'xtick.labelsize': 20,      # X轴刻度标签大小
#     'ytick.labelsize': 20,      # Y轴刻度标签大小
#     'legend.fontsize': 18,     # 图例字体大小
# })
#
#
#
# # 示例数据
# groups = [ '42-58', '59-63', '64-67', '68-74']  # 组别
# means_left = [ 55.0, 48.6, 52.9, 58.3]  # 左柱均值
# stds_left = [ 19.2, 21.0, 22.0, 22.9]  # 左柱标准差
# means_right = [ 42.9, 47.1, 40.1, 44.9]  # 右柱均值
# stds_right = [ 18.3, 15.5, 19.9, 19.0]  # 右柱标准差
#
# # 可调整参数
# bar_width = 0.45  # 单根柱子的宽度
# group_spacing = 1.45  # 组间距离 (1.0是默认值，增大此值会增加组间距离)
#
# x = np.arange(len(groups)) * group_spacing  # 组的x位置，乘以组间距离系数
# width = bar_width  # 柱子的宽度
#
# fig, ax = plt.subplots(figsize=(8.9, 7))  # 进一步增加图形大小
# cmap = plt.get_cmap("coolwarm")
# # color1 = cmap(0.15)    # 最冷色（深蓝）
# # color2 = cmap(0.85)    # 最暖色（深红）
# color1 = '#AAC4FE'    #
# color2 = '#F18E70'   #
#
# # 绘制柱状图
# left_bars = ax.bar(x - width/2, means_left, width, yerr=stds_left, label='Lower', capsize=5,color = color1, error_kw={
#         'elinewidth': 1,    # 误差条主线的宽度
#         'capthick': 1,      # 误差条横线的厚度
#         'ecolor': 'black',   # 误差条颜色
#     })
# right_bars = ax.bar(x + width/2, means_right, width, yerr=stds_right, label='Higher',color = color2, capsize=5,error_kw={
#         'elinewidth': 1,    # 误差条主线的宽度
#         'capthick': 1,      # 误差条横线的厚度
#         'ecolor': 'black',   # 误差条颜色
#     })
#
# # 添加文本标签
# def add_labels_left(bars, stds):
#     for idx, (bar, std) in enumerate(zip(bars, stds)):
#         if(idx==1):
#             height = bar.get_height()
#             ax.text(bar.get_x() +0.08 + bar.get_width()/2., height + 5,
#                 f'{height:.1f}±{std:.1f}',
#                 ha='center', va='bottom', fontsize=16)
#         else:
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.08 + bar.get_width() / 2., height + 1,
#                     f'{height:.1f}±{std:.1f}',
#                     ha='center', va='bottom', fontsize=16)
#
# def add_labels_right(bars, stds):
#     for bar, std in zip(bars, stds):
#
#             height = bar.get_height()
#             ax.text(bar.get_x() - 0.09 + bar.get_width()/2., height + 1,
#                 f'{height:.1f}±{std:.1f}',
#                 ha='center', va='bottom', fontsize=16)
#
#
# add_labels_left(left_bars, stds_left)
# add_labels_right(right_bars, stds_right)
#
# # 添加组间比较的完整括号和星号
# def add_significance_brackets(ax, x1, x2, y, text, line_height=3):
#     # 绘制完整的括号形状（水平线+两端竖线）
#     bracket_height = line_height
#     # 水平线
#     ax.plot([x1, x2], [y-2.5, y-2.5], lw=1, color='k')
#     # 左端竖线
#     ax.plot([x1, x1], [y-2.5, y-bracket_height-2.5], lw=1, color='k')
#     # 右端竖线
#     ax.plot([x2, x2], [y-2.5, y-bracket_height-2.5], lw=1, color='k')
#     # 添加星号文本（位置在水平线上方）
#     ax.text((x1+x2)*0.5, y + bracket_height*0.2-3.1, text,
#             ha='center', va='bottom', fontsize=16)
#
# # 假设的显著性结果 (这里需要替换为你的实际统计结果)
# # 格式: (group_index, y_position, significance_text)
# significance_data = [
#     (0, 80, '*'),    # 第3组显著
#     (1, 80, ''),    # 第4组显著
#     (2, 80, '*'),     # 第5组不显著
#     (3, 80, '**'),    # 第6组显著
#
# ]
#
# for i, y, text in significance_data:
#     if((i!=1)&(i!=1)):
#         x1 = x[i] - width/2 - 0.1
#         x2 = x[i] + width/2 + 0.1
#         add_significance_brackets(ax, x1, x2, 90, text)
#
# ax.set_ylim(0, 110)  # 调整y轴范围以容纳显著性标记
# ax.set_xticks(x)
# ax.set_xticklabels(groups)
# ax.set_ylabel('Calibrated FEV$_{1}$%pred')
# ax.set_xlabel('Age group')
# ax.set_title('')
# ax.legend()
#
# plt.tight_layout()
# # plt.show()
# output_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/test-copd真实编年相关性图/external1.png'
# plt.savefig(output_path, bbox_inches='tight', dpi=600)






#
#
#
# # # ###########新的pred差异-------external2
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置全局字体大小（基准值，可被局部覆盖）
# rcParams.update({
#     'figure.dpi': 600,  # 降低 DPI 以避免显示问题
#     'font.size': 20,           # 默认字体大小（影响大多数文本）
#     'axes.titlesize': 20,      # 标题字体大小
#     'axes.labelsize': 20,      # 坐标轴标签大小
#     'xtick.labelsize': 20,      # X轴刻度标签大小
#     'ytick.labelsize': 20,      # Y轴刻度标签大小
#     'legend.fontsize': 18,     # 图例字体大小
# })
#
#
#
# # 示例数据
# groups = ['36-65', '66-74' ]  # 组别
# means_left = [61.4, 68.8]  # 左柱均值
# stds_left = [22.0, 15.0]  # 左柱标准差
# means_right = [49.0, 43.2]  # 右柱均值
# stds_right = [15.3, 12.6]  # 右柱标准差
#
# # 可调整参数
# bar_width = 0.55  # 单根柱子的宽度
# group_spacing = 1.7  # 组间距离 (1.0是默认值，增大此值会增加组间距离)
#
# x = np.arange(len(groups)) * group_spacing  # 组的x位置，乘以组间距离系数
# width = bar_width  # 柱子的宽度
#
# fig, ax = plt.subplots(figsize=(5.2, 7))  # 进一步增加图形大小
# cmap = plt.get_cmap("coolwarm")
# # color1 = cmap(0.15)    # 最冷色（深蓝）
# # color2 = cmap(0.85)    # 最暖色（深红）
# color1 = '#AAC4FE'    #
# color2 = '#F18E70'   #
# # 绘制柱状图
# left_bars = ax.bar(x - width/2, means_left, width, yerr=stds_left, label='Lower', capsize=5,color = color1, error_kw={
#         'elinewidth': 1,    # 误差条主线的宽度
#         'capthick': 1,      # 误差条横线的厚度
#         'ecolor': 'black'   # 误差条颜色
#     })
# right_bars = ax.bar(x + width/2, means_right, width, yerr=stds_right, label='Higher', capsize=5,color = color2, error_kw={
#         'elinewidth': 1,    # 误差条主线的宽度
#         'capthick': 1,      # 误差条横线的厚度
#         'ecolor': 'black'   # 误差条颜色
#     })
#
# # 添加文本标签
# def add_labels_left(bars, stds):
#     for bar, std in zip(bars, stds):
#         height = bar.get_height()
#         ax.text(bar.get_x() -0.1 + bar.get_width()/2., height + 1,
#                 f'{height:.1f}±{std:.1f}',
#                 ha='center', va='bottom', fontsize=16)
# def add_labels_right(bars, stds):
#     for bar, std in zip(bars, stds):
#         height = bar.get_height()
#         ax.text(bar.get_x() + 0.1 + bar.get_width()/2., height + 1,
#                 f'{height:.1f}±{std:.1f}',
#                 ha='center', va='bottom', fontsize=16)
#
# add_labels_left(left_bars, stds_left)
# add_labels_right(right_bars, stds_right)
#
# # 添加组间比较的完整括号和星号
# def add_significance_brackets(ax, x1, x2, y, text, line_height=3):
#     # 绘制完整的括号形状（水平线+两端竖线）
#     bracket_height = line_height
#     # 水平线
#     ax.plot([x1, x2], [y-2.5, y-2.5], lw=1, color='k')
#     # 左端竖线
#     ax.plot([x1, x1], [y-2.5, y-bracket_height-2.5], lw=1, color='k')
#     # 右端竖线
#     ax.plot([x2, x2], [y-2.5, y-bracket_height-2.5], lw=1, color='k')
#     # 添加星号文本（位置在水平线上方）
#     ax.text((x1+x2)*0.5, y + bracket_height*0.2-3.1, text,
#             ha='center', va='bottom', fontsize=16)
#
# # 假设的显著性结果 (这里需要替换为你的实际统计结果)
# # 格式: (group_index, y_position, significance_text)
# significance_data = [
#     (0, 80, '*'),    # 第1组显著
#     (1, 80, '***')    # 第2组显著
#
#
# ]
#
# for i, y, text in significance_data:
#     if((i!=2)&(i!=3)):
#         x1 = x[i] - width/2 - 0.1
#         x2 = x[i] + width/2 + 0.1
#         add_significance_brackets(ax, x1, x2, 90, text)
#
# ax.set_ylim(0, 110)  # 调整y轴范围以容纳显著性标记
# ax.set_xticks(x)
# ax.set_xticklabels(groups)
# ax.set_ylabel('Calibrated FEV$_{1}$%pred')
# ax.set_xlabel('Age group')
# ax.set_title('')
# ax.legend()
# ax.set_xlim(x[0] - 1.0, x[-1] + 1.0)  # 左右各扩展 1.0 单位
#
# plt.tight_layout()
# # plt.show()
# output_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/test-copd真实编年相关性图/external2.png'
# plt.savefig(output_path, bbox_inches='tight', dpi=600)









# # ###########新的pred差异-------external3
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置全局字体大小（基准值，可被局部覆盖）
# rcParams.update({
#     'figure.dpi': 600,  # 降低 DPI 以避免显示问题
#     'font.size': 20,           # 默认字体大小（影响大多数文本）
#     'axes.titlesize': 20,      # 标题字体大小
#     'axes.labelsize': 20,      # 坐标轴标签大小
#     'xtick.labelsize': 20,      # X轴刻度标签大小
#     'ytick.labelsize': 20,      # Y轴刻度标签大小
#     'legend.fontsize': 18,     # 图例字体大小
# })
#
#
#
# # 示例数据
# groups = ['45-60', '61-65', '66-74' ]  # 组别
# means_left = [63.9, 58.0, 68.4]  # 左柱均值
# stds_left = [19.7, 17.2, 18.1]  # 左柱标准差
# means_right = [49.9, 55.6, 53.9]  # 右柱均值
# stds_right = [19.4, 16.3, 19.5]  # 右柱标准差
#
# # 可调整参数
# bar_width = 0.52  # 单根柱子的宽度
# group_spacing = 1.6  # 组间距离 (1.0是默认值，增大此值会增加组间距离)
#
# x = np.arange(len(groups)) * group_spacing  # 组的x位置，乘以组间距离系数
# width = bar_width  # 柱子的宽度
#
# fig, ax = plt.subplots(figsize=(7.5, 7))  # 进一步增加图形大小
#
# cmap = plt.get_cmap("coolwarm")
# # color1 = cmap(0.15)    # 最冷色（深蓝）
# # color2 = cmap(0.85)    # 最暖色（深红）
# color1 = '#AAC4FE'    #
# color2 = '#F18E70'   #
# # 绘制柱状图
# left_bars = ax.bar(x - width/2, means_left, width, yerr=stds_left, label='Lower', capsize=5,color = color1, error_kw={
#         'elinewidth': 1,    # 误差条主线的宽度
#         'capthick': 1,      # 误差条横线的厚度
#         'ecolor': 'black'   # 误差条颜色
#     })
# right_bars = ax.bar(x + width/2, means_right, width, yerr=stds_right, label='Higher', capsize=5,color = color2, error_kw={
#         'elinewidth': 1,    # 误差条主线的宽度
#         'capthick': 1,      # 误差条横线的厚度
#         'ecolor': 'black'   # 误差条颜色
#     })
#
# # 添加文本标签
# def add_labels_left(bars, stds):
#     for idx,(bar, std) in enumerate(zip(bars, stds)):
#         if(idx==1):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.21 + bar.get_width()/2., height + 5,
#                 f'{height:.1f}±{std:.1f}',
#                 ha='center', va='bottom', fontsize=16)
#         else:
#             height = bar.get_height()
#             ax.text(bar.get_x() - 0.21 + bar.get_width() / 2., height + 1,
#                     f'{height:.1f}±{std:.1f}',
#                     ha='center', va='bottom', fontsize=16)
# def add_labels_right(bars, stds):
#     for bar, std in zip(bars, stds):
#         height = bar.get_height()
#         ax.text(bar.get_x() + 0.2 + bar.get_width()/2., height + 1,
#                 f'{height:.1f}±{std:.1f}',
#                 ha='center', va='bottom', fontsize=16)
#
# add_labels_left(left_bars, stds_left)
# add_labels_right(right_bars, stds_right)
#
# # 添加组间比较的完整括号和星号
# def add_significance_brackets(ax, x1, x2, y, text, line_height=3):
#     # 绘制完整的括号形状（水平线+两端竖线）
#     bracket_height = line_height
#     # 水平线
#     ax.plot([x1, x2], [y-2.5, y-2.5], lw=1, color='k')
#     # 左端竖线
#     ax.plot([x1, x1], [y-2.5, y-bracket_height-2.5], lw=1, color='k')
#     # 右端竖线
#     ax.plot([x2, x2], [y-2.5, y-bracket_height-2.5], lw=1, color='k')
#     # 添加星号文本（位置在水平线上方）
#     ax.text((x1+x2)*0.5, y + bracket_height*0.05-3.1, text,
#             ha='center', va='bottom', fontsize=16)
#
# # 假设的显著性结果 (这里需要替换为你的实际统计结果)
# # 格式: (group_index, y_position, significance_text)
# significance_data = [
#     (0, 80, '*'),    # 第1组显著
#     (1, 80, ''),    # 第2组显著
#     (2, 80, '*')  # 第2组显著
#
# ]
#
# for i, y, text in significance_data:
#     if((i!=1)&(i!=1)):
#         x1 = x[i] - width/2 - 0.1
#         x2 = x[i] + width/2 + 0.1
#         add_significance_brackets(ax, x1, x2, 90, text)
#
#
# ax.set_ylim(0, 110)  # 调整y轴范围以容纳显著性标记
# ax.set_xticks(x)
# ax.set_xticklabels(groups)
# ax.set_ylabel('Calibrated FEV$_{1}$%pred')
# ax.set_xlabel('Age group')
# ax.set_title('')
# ax.legend()
# ax.set_xlim(x[0] - 1.0, x[-1] + 1.0)  # 左右各扩展 1.0 单位
#
# plt.tight_layout()
# # plt.show()
# output_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/test-copd真实编年相关性图/external3.png'
# plt.savefig(output_path, bbox_inches='tight', dpi=600)









# # # # #######其余相关的
# # # # #########散点校正PRED F/F
# # # # # ############散点图1   FEV1--MAE
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, gaussian_kde
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
#
# from matplotlib import rcParams
# def average_every_n(data, n=2):
#     """接收一个 Series 并返回一个新的 Series，其中每 n 个元素的平均值"""
#     return data.groupby(data.index // n).mean()
# rcParams.update({
#     'figure.dpi': 100,
#
# })
# # 读取 Excel 文件
# file_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/test-copd真实编年相关性图/copd.xlsx'  # 替换为你的文件路径
# data = pd.read_excel(file_path)
#
# # 假设第一列是 y（纵轴数据），第二列是 x（横轴数据）
# y = data.iloc[:, 0]  # 第一列
# x = data.iloc[:, 1]  # 第二列
#
# # 设置 n 的值
# n = 1  # 你可以根据需要更改这个值
#
# # 对数据进行处理：每 n 个数据取平均值
# y_avg = average_every_n(y, n)
# x_avg = average_every_n(x, n)
#
# # 计算 Pearson 相关系数 r 和 P 值
# r, p_value = pearsonr(x_avg, y_avg)
#
# # 计算样本整体的回归斜率
# slope, _ = np.polyfit(x_avg, y_avg, 1)
#
# # 计算高斯核密度估计，确定散点图中密度最高的点
# xy = np.vstack([x_avg, y_avg])
# kde = gaussian_kde(xy)
# # 构造网格
# x_grid = np.linspace(x_avg.min(), x_avg.max(), 100)
# y_grid = np.linspace(y_avg.min(), y_avg.max(), 100)
# X, Y = np.meshgrid(x_grid, y_grid)
# Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
# max_index = np.unravel_index(np.argmax(Z), Z.shape)
# densest_x = X[max_index]
# densest_y = Y[max_index]
#
#
# # 计算密度值（每个点的）
# xy_sample = np.vstack([x_avg, y_avg])
# point_densities = kde(xy_sample)
#
# # 绘制打点，点颜色表示密度（渐变色效果）
# # plt.figure(figsize=(13.5,12))
# plt.figure(figsize=(16,16))
# ax = plt.gca()
# color_values = x_grid + y_grid  # 可以根据需要更改此计算函数
#
# # 使用"coolwarm" colormap（从红色到蓝色）来实现中心红色，外部蓝色的渐变
# cmap = plt.get_cmap("coolwarm")
# norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
# sm = ScalarMappable(cmap=cmap, norm=norm)
#
# sc = plt.scatter(x_avg, y_avg, c=point_densities, cmap="coolwarm", s=200, alpha=0.8)
#
# # 添加颜色条
# # plt.colorbar(sc, label='Density')
#
# # 获取坐标轴范围
# xlims = ax.get_xlim()
#
# # 绘制密度最高点为中心的回归线
# x_line = np.array(xlims)
# y_line = slope * (x_line - densest_x) + densest_y
# plt.plot(x_line, y_line, color='black', lw=4, linestyle='-')
#
#
# # 设定标题和标签
# # 设定标题和标签（y 控制标题垂直位置）
#
# plt.xlabel("Predicted Age", fontsize=45)
# #
# #Lung age gap
# #
# #Calibrated Lung age gap
# plt.ylabel("Chronological Age", fontsize=45)
# #Calibrated FEV1/FEV1_Pred
# #FEV1/FEV1_Pred
# #FEV1/FVC_x
# #Calibrated FEV1
# #Calibrated FVC
# #Calibrated FEV1/FVC_x
#
#
#
# plt.tick_params(axis='x', labelsize=45)
# plt.tick_params(axis='y', labelsize=45)
#
# plt.title(f"r = {r:.2f}, p-value = {p_value:.2e}", y=0.90, fontsize = 45)  # 将标题位置向下移动
# # plt.ylim(1,120)
# # plt.xlim(30,80)
# # plt.ylim(0.11,0.8)
# ######Lung age gap
# # plt.ylim(0.21,0.8)
# # plt.ylim(1,120)
# # # plt.xlim(-15,15)
# plt.ylim(30, 88)  # y 轴范围从 0 到 6（确保 0-1.5 的空间存在）
# plt.yticks(np.arange(40, 81, 10))  #
#
# plt.xlim(30,80)
# plt.xticks(np.arange(30, 81, 10))
# #######FVC
# # plt.ylim(0.8,5.5)
# # plt.xlim(30,90)
# # plt.xlim(-15,15)
# #####FEV1
# # plt.ylim(0.1,3.8)
# # plt.ylim(0.1,6)
# # plt.xlim(30,80)
# # plt.xlim(-15,15)
# ######f/f
# # plt.ylim(0.21,0.9)
# # plt.ylim(0.21,0.8)
# # plt.ylim(0.61,1.1)
# # plt.xlim(30,80)
# # 手动设置 x 轴刻度（例如每 10 个单位显示一个刻度）
# # plt.xticks(np.arange(-15, 16, 5))  # -15  -10  -5  0  5  10  15
# # plt.xticks(np.arange(-30, 31, 10))  # -15  -10  -5  0  5  10  15
# # plt.xticks(np.arange(-25, 26, 10))  #
# # plt.xticks(np.arange(-20, 21, 5))  #
# # plt.xticks(np.arange(30, 81, 10))  # 30, 40, 50, 60, 70, 80
# # plt.yticks(np.arange(0.1, 3.8, 6))  # 30, 40, 50, 60, 70, 80
#
# #
# # plt.ylim(0.4, 5.5)  # y 轴范围从 0 到 6（确保 0-1.5 的空间存在）
# # plt.yticks(np.arange(1.5, 5.6, 1))  #
#
# # plt.yticks([1,1.9,2.8,3.7,4.6,5.5])
# # 保存图片到指定路径
# output_path = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/test-copd真实编年相关性图/copd.png'
# plt.savefig(output_path, bbox_inches='tight', dpi=600)
#
# # plt.show()







#
# ##########PRED-emphysema index（气肿指数）
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置全局字体大小（基准值，可被局部覆盖）
# rcParams.update({
#     'figure.dpi': 600,  # 降低 DPI 以避免显示问题
#     'font.size': 18,           # 默认字体大小（影响大多数文本）
#     'axes.titlesize': 18,      # 标题字体大小
#     'axes.labelsize': 18,      # 坐标轴标签大小
#     'xtick.labelsize': 18,      # X轴刻度标签大小
#     'ytick.labelsize': 18,      # Y轴刻度标签大小
#     'legend.fontsize': 16     # 图例字体大小
# })
# # 示例数据
# groups = ['35-43', '44-50', '51-53', '54-56', '57-60', '61-65', '66-74']  # 组别
# means_right = [0.169, 0.157, 0.177, 0.160, 0.168, 0.190, 0.179]  # 左柱均值
# stds_right = [0.060, 0.054, 0.046, 0.066, 0.065, 0.061, 0.066]  # 左柱标准差
# means_left = [0.166, 0.152, 0.129, 0.139, 0.143, 0.151, 0.172]  # 右柱均值
# stds_left = [0.049, 0.058, 0.061, 0.056, 0.061, 0.064, 0.056]  # 右柱标准差
#
# # 可调整参数
# bar_width = 0.6  # 单根柱子的宽度
# group_spacing = 1.7  # 组间距离 (1.0是默认值，增大此值会增加组间距离)
#
# x = np.arange(len(groups)) * group_spacing  # 组的x位置，乘以组间距离系数
# width = bar_width  # 柱子的宽度
#
# fig, ax = plt.subplots(figsize=(12, 7))  # 进一步增加图形大小
#
#
# ########
# cmap = plt.get_cmap("coolwarm")
# # color1 = cmap(0.15)    # 最冷色（深蓝）
# # color2 = cmap(0.85)    # 最暖色（深红）
# color1 = '#C8D7EB'    #
# color2 = '#FAEBC7'    #
# # 绘制柱状图
# left_bars = ax.bar(x - width/2, means_left, width, yerr=stds_left, label='Lower', capsize=5, color = color1)
# right_bars = ax.bar(x + width/2, means_right, width, yerr=stds_right, label='Higher', capsize=5, color = color2)
#
# # 添加文本标签
# # def add_labels(bars, stds):
# #     for bar, std in zip(bars, stds):
# #         height = bar.get_height()
# #         ax.text(bar.get_x() + bar.get_width()/2., height + 1,
# #                 f'{height:.1f}±{std:.1f}',
# #                 ha='center', va='bottom', fontsize=15)
# #
# # add_labels(left_bars, stds_left)
# # add_labels(right_bars, stds_right)
#
# # # 添加文本标签
# def add_labels_left(bars, stds):
#     i=0
#     for bar, std in zip(bars, stds):
#         i=i+1
#         if((i==8)):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.135 + bar.get_width()/2., height+0.005,
#                 f'{height:.3f}±{std:.2f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif((i==28)):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.135 + bar.get_width()/2., height+0.007,
#                 f'{height:.3f}±{std:.2f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif(i==8):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.135 + bar.get_width()/2., height+0.0155,
#                 f'{height:.3f}±{std:.2f}',
#                 ha='center', va='bottom', fontsize=14)
#         else:
#             height = bar.get_height()
#             ax.text(bar.get_x() - 0.135 + bar.get_width() / 2., height,
#                     f'{height:.3f}±{std:.2f}',
#                     ha='center', va='bottom', fontsize=14)
# def add_labels_right(bars, stds):
#     i = 0
#     for bar, std in zip(bars, stds):
#         i=i+1
#         if(i==7):
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.172 + bar.get_width()/2., height+0.004,
#                 f'{height:.3f}±{std:.2f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif(i==1):
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.172 + bar.get_width()/2., height+0.006,
#                 f'{height:.3f}±{std:.2f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif (i == 2):
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.172 + bar.get_width() / 2., height + 0.004,
#                     f'{height:.3f}±{std:.2f}',
#                     ha='center', va='bottom', fontsize=14)
#         else:
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.172 + bar.get_width() / 2., height,
#                     f'{height:.3f}±{std:.2f}',
#                     ha='center', va='bottom', fontsize=14)
#
# add_labels_left(left_bars, stds_left)
# add_labels_right(right_bars, stds_right)
#
# # 添加组间比较的完整括号和星号
# def add_significance_brackets(ax, x1, x2, y, text, line_height=0.01):
#     # 绘制完整的括号形状（水平线+两端竖线）
#     bracket_height = line_height
#     # 水平线
#     ax.plot([x1, x2], [y, y], lw=1, color='k')
#     # 左端竖线
#     ax.plot([x1, x1], [y, y-bracket_height], lw=1, color='k')
#     # 右端竖线
#     ax.plot([x2, x2], [y, y-bracket_height], lw=1, color='k')
#     # 添加星号文本（位置在水平线上方）
#     ax.text((x1+x2)*0.5, y + bracket_height*0.3, text,
#             ha='center', va='bottom', fontsize=16)
#
# # 假设的显著性结果 (这里需要替换为你的实际统计结果)
# # 格式: (group_index, y_position, significance_text)
# significance_data = [
#     (0, 0.27, ''),    # 第1组显著
#     (1, 0.27, ''),    # 第2组显著
#     (2, 0.27, '**'),    # 第3组显著
#     (3, 0.27, ''),    # 第4组显著
#     (4, 0.27, '*'),     # 第5组不显著
#     (5, 0.27, '*'),    # 第6组显著
#     (6, 0.27, ''),     # 第7组不显著
# ]
#
# for i, y, text in significance_data:
#     if((i!=0)&(i!=1)&(i!=3)&(i!=6)):
#         x1 = x[i] - width/2 - 0.1
#         x2 = x[i] + width/2 + 0.1
#         add_significance_brackets(ax, x1, x2, y, text)
#
#
# plt.ylim(0, 0.3)  # 调整y轴范围以容纳显著性标记
# plt.yticks(np.arange(0, 0.3, 0.05))  # 30, 40, 50, 60, 70, 80
#
# ax.set_xticks(x)
# ax.set_xticklabels(groups)
# ax.set_ylabel('All emphysema index')
# ax.set_xlabel('Age group')
# ax.set_title('')
# ax.legend()
#
# plt.tight_layout()
# # plt.show()
# output_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/第一步/resnet34/resnet_34/气肿指数.png'
# plt.savefig(output_path, bbox_inches='tight', dpi=600)









# # ##########Perc 15(绝对值)
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置全局字体大小（基准值，可被局部覆盖）
# rcParams.update({
#     'figure.dpi': 600,  # 降低 DPI 以避免显示问题
#     'font.size': 18,           # 默认字体大小（影响大多数文本）
#     'axes.titlesize': 18,      # 标题字体大小
#     'axes.labelsize': 18,      # 坐标轴标签大小
#     'xtick.labelsize': 18,      # X轴刻度标签大小
#     'ytick.labelsize': 18,      # Y轴刻度标签大小
#     'legend.fontsize': 16     # 图例字体大小
# })
# # 示例数据
# groups = ['35-43', '44-50', '51-53', '54-56', '57-60', '61-65', '66-74']  # 组别
# means_right = [953.4, 948.4, 957.8, 948.4, 951.5, 960.9, 957.4]  # 左柱均值
# stds_right = [21.2, 23.6, 13.2, 26.5, 26.1, 18.1, 20.9]  # 左柱标准差
# means_left = [952.6, 946.5, 936.7, 940.2, 940.6, 945.5, 953.2]  # 右柱均值
# stds_left = [19.1, 23.5, 26.0, 25.9, 27.7, 24.9, 23.3]  # 右柱标准差
#
# # 可调整参数
# bar_width = 0.6  # 单根柱子的宽度
# group_spacing = 1.7  # 组间距离 (1.0是默认值，增大此值会增加组间距离)
#
# x = np.arange(len(groups)) * group_spacing  # 组的x位置，乘以组间距离系数
# width = bar_width  # 柱子的宽度
#
# fig, ax = plt.subplots(figsize=(12, 7))  # 进一步增加图形大小
#
#
# ########
# cmap = plt.get_cmap("coolwarm")
# # color1 = cmap(0.15)    # 最冷色（深蓝）
# # color2 = cmap(0.85)    # 最暖色（深红）
# # color1 = '#EFD496'    #
# # color2 = '#CDE0C7'    #
# color1 = '#C8D7EB'    #
# color2 = '#FAEBC7'    #
# # 绘制柱状图
# left_bars = ax.bar(x - width/2, means_left, width, yerr=stds_left, label='Lower', capsize=5, color = color1)
# right_bars = ax.bar(x + width/2, means_right, width, yerr=stds_right, label='Higher', capsize=5, color = color2)
#
# # 添加文本标签
# # def add_labels(bars, stds):
# #     for bar, std in zip(bars, stds):
# #         height = bar.get_height()
# #         ax.text(bar.get_x() + bar.get_width()/2., height + 1,
# #                 f'{height:.1f}±{std:.1f}',
# #                 ha='center', va='bottom', fontsize=15)
# #
# # add_labels(left_bars, stds_left)
# # add_labels(right_bars, stds_right)
#
# # # 添加文本标签
# def add_labels_left(bars, stds):
#     i=0
#     for bar, std in zip(bars, stds):
#         i=i+1
#         if((i==8)):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.135 + bar.get_width()/2., height+0.005,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif((i==8)):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.135 + bar.get_width()/2., height+0.007,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif(i==8):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.135 + bar.get_width()/2., height+3,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         else:
#             height = bar.get_height()
#             ax.text(bar.get_x() - 0.135 + bar.get_width() / 2., height,
#                     f'{height:.0f}±{std:.0f}',
#                     ha='center', va='bottom', fontsize=14)
# def add_labels_right(bars, stds):
#     i = 0
#     for bar, std in zip(bars, stds):
#         i=i+1
#         if(i==1):
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.172 + bar.get_width()/2., height+1.5,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         else:
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.172 + bar.get_width() / 2., height,
#                     f'{height:.0f}±{std:.0f}',
#                     ha='center', va='bottom', fontsize=14)
#
# add_labels_left(left_bars, stds_left)
# add_labels_right(right_bars, stds_right)
#
# # 添加组间比较的完整括号和星号
# def add_significance_brackets(ax, x1, x2, y, text, line_height=980):
#     # 绘制完整的括号形状（水平线+两端竖线）
#     bracket_height = line_height
#     # 水平线
#     ax.plot([x1, x2], [y, y], lw=1, color='k')
#     # 左端竖线
#     ax.plot([x1, x1], [985,982], lw=1, color='k')
#     # 右端竖线
#     ax.plot([x2, x2], [985,982], lw=1, color='k')
#     # 添加星号文本（位置在水平线上方）
#     ax.text((x1+x2)*0.5, y + 1, text,
#             ha='center', va='bottom', fontsize=16)
#
# # 假设的显著性结果 (这里需要替换为你的实际统计结果)
# # 格式: (group_index, y_position, significance_text)
# significance_data = [
#     (0, 985, ''),    # 第1组显著
#     (1, 985, ''),    # 第2组显著
#     (2, 985, '***'),    # 第3组显著
#     (3, 985, ''),    # 第4组显著
#     (4, 985, '*'),     # 第5组不显著
#     (5, 985, '**'),    # 第6组显著
#     (6, 985, ''),     # 第7组不显著
# ]
#
# for i, y, text in significance_data:
#     if((i!=0)&(i!=1)&(i!=3)&(i!=6)):
#         x1 = x[i] - width/2 - 0.1
#         x2 = x[i] + width/2 + 0.1
#         add_significance_brackets(ax, x1, x2, y, text)
#
#
# plt.ylim(900, 1000)  # 调整y轴范围以容纳显著性标记
# plt.yticks(np.arange(900, 1001, 20))  # 30, 40, 50, 60, 70, 80
#
# ax.set_xticks(x)
# ax.set_xticklabels(groups)
# ax.set_ylabel('|Perc 15|')
# ax.set_xlabel('Age group')
# ax.set_title('')
# ax.legend()
#
# plt.tight_layout()
# # plt.show()
# output_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/第一步/resnet34/resnet_34/perc15.png'
# plt.savefig(output_path, bbox_inches='tight', dpi=600)






# ##########Perc 15（正常负值）
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置全局字体大小（基准值，可被局部覆盖）
# rcParams.update({
#     'figure.dpi':600,  # 降低 DPI 以避免显示问题
#     'font.size': 18,           # 默认字体大小（影响大多数文本）
#     'axes.titlesize': 18,      # 标题字体大小
#     'axes.labelsize': 18,      # 坐标轴标签大小
#     'xtick.labelsize': 18,      # X轴刻度标签大小
#     'ytick.labelsize': 18,      # Y轴刻度标签大小
#     'legend.fontsize': 16     # 图例字体大小
# })
# # 示例数据
# groups = ['35-43', '44-50', '51-53', '54-56', '57-60', '61-65', '66-74']  # 组别
# means_right = [-953.4, -948.4, -957.8, -948.4, -951.5, -960.9, -957.4]  # 左柱均值
# stds_right = [21.2, 23.6, 13.2, 26.5, 26.1, 18.1, 20.9]  # 左柱标准差
# means_left = [-952.6, -946.5, -936.7, -940.2, -940.6, -945.5, -953.2]  # 右柱均值
# stds_left = [19.1, 23.5, 26.0, 25.9, 27.7, 24.9, 23.3]  # 右柱标准差
#
# # 可调整参数
# bar_width = 0.6  # 单根柱子的宽度
# group_spacing = 1.7  # 组间距离 (1.0是默认值，增大此值会增加组间距离)
#
# x = np.arange(len(groups)) * group_spacing  # 组的x位置，乘以组间距离系数
# width = bar_width  # 柱子的宽度
#
# fig, ax = plt.subplots(figsize=(12, 7))  # 进一步增加图形大小
#
#
# ########
# cmap = plt.get_cmap("coolwarm")
# # color1 = cmap(0.15)    # 最冷色（深蓝）
# # color2 = cmap(0.85)    # 最暖色（深红）
# # color1 = '#EFD496'    #
# # color2 = '#CDE0C7'    #
# color1 = '#C8D7EB'    #
# color2 = '#FAEBC7'    #
# # 绘制柱状图
# left_bars = ax.bar(x - width/2, means_left, width, yerr=stds_left, label='Lower', capsize=5, color = color1)
# right_bars = ax.bar(x + width/2, means_right, width, yerr=stds_right, label='Higher', capsize=5, color = color2)
#
# # 添加文本标签
# # def add_labels(bars, stds):
# #     for bar, std in zip(bars, stds):
# #         height = bar.get_height()
# #         ax.text(bar.get_x() + bar.get_width()/2., height + 1,
# #                 f'{height:.1f}±{std:.1f}',
# #                 ha='center', va='bottom', fontsize=15)
# #
# # add_labels(left_bars, stds_left)
# # add_labels(right_bars, stds_right)
#
# # # 添加文本标签
# def add_labels_left(bars, stds):
#     i=0
#     for bar, std in zip(bars, stds):
#         i=i+1
#         if((i==8)):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.135 + bar.get_width()/2., height+0.005,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif((i==8)):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.135 + bar.get_width()/2., height+0.007,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif(i==8):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.135 + bar.get_width()/2., height+3,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         else:
#             height = bar.get_height()
#             ax.text(bar.get_x() - 0.235 + bar.get_width() / 2., height,
#                     f'{height:.0f}±{std:.0f}',
#                     ha='center', va='bottom', fontsize=14)
# def add_labels_right(bars, stds):
#     i = 0
#     for bar, std in zip(bars, stds):
#         i=i+1
#         if(i==1):
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.16 + bar.get_width()/2., height-2.5,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif(i==2):
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.16 + bar.get_width()/2., height-1,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         else:
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.16 + bar.get_width() / 2., height,
#                     f'{height:.0f}±{std:.0f}',
#                     ha='center', va='bottom', fontsize=14)
#
# add_labels_left(left_bars, stds_left)
# add_labels_right(right_bars, stds_right)
#
# # 添加组间比较的完整括号和星号
# def add_significance_brackets(ax, x1, x2, y, text, line_height=-980):
#     # 绘制完整的括号形状（水平线+两端竖线）
#     bracket_height = line_height
#     # 水平线
#     ax.plot([x1, x2], [y, y], lw=1, color='k')
#     # 左端竖线
#     ax.plot([x1, x1], [-985,-982], lw=1, color='k')
#     # 右端竖线
#     ax.plot([x2, x2], [-985,-982], lw=1, color='k')
#     # 添加星号文本（位置在水平线上方）
#     ax.text((x1+x2)*0.5, y + 1, text,
#             ha='center', va='bottom', fontsize=16)
#
# # 假设的显著性结果 (这里需要替换为你的实际统计结果)
# # 格式: (group_index, y_position, significance_text)
# significance_data = [
#     (0, -985, ''),    # 第1组显著
#     (1, -985, ''),    # 第2组显著
#     (2, -985, '***'),    # 第3组显著
#     (3, -985, ''),    # 第4组显著
#     (4, -985, '*'),     # 第5组不显著
#     (5, -985, '**'),    # 第6组显著
#     (6, -985, ''),     # 第7组不显著
# ]
#
# for i, y, text in significance_data:
#     if((i!=0)&(i!=1)&(i!=3)&(i!=6)):
#         x1 = x[i] - width/2 - 0.1
#         x2 = x[i] + width/2 + 0.1
#         add_significance_brackets(ax, x1, x2, y, text)
#
#
# plt.ylim(-900, -1000)  # 调整y轴范围以容纳显著性标记
# plt.yticks(np.arange(-900, -1000, -20))  # 30, 40, 50, 60, 70, 80
#
# ax.set_xticks(x)
# ax.set_xticklabels(groups)
# ax.set_ylabel('Perc 15')
# ax.set_xlabel('Age group')
# ax.set_title('')
# ax.legend()
#
# plt.tight_layout()
# # plt.show()
# output_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/第一步/resnet34/resnet_34/perc15.png'
# plt.savefig(output_path, bbox_inches='tight', dpi=600)








# #################肺密度
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # 设置全局字体大小（基准值，可被局部覆盖）
# rcParams.update({
#     'figure.dpi': 600,  # 降低 DPI 以避免显示问题
#     'font.size': 18,           # 默认字体大小（影响大多数文本）
#     'axes.titlesize': 18,      # 标题字体大小
#     'axes.labelsize': 18,      # 坐标轴标签大小
#     'xtick.labelsize': 18,      # X轴刻度标签大小
#     'ytick.labelsize': 18,      # Y轴刻度标签大小
#     'legend.fontsize': 16     # 图例字体大小
# })
# # 示例数据
# groups = ['35-43', '44-50', '51-53', '54-56', '57-60', '61-65', '66-74']  # 组别
# means_right = [-832.1, -821.8, -837.1, -823.4, -823.4, -835.8, -825.5]  # 左柱均值
# stds_right = [28.4, 37.4, 19.5, 37.1, 39.2, 22.4, 25.5]  # 左柱标准差
# means_left = [-832.4, -820.5, -807.8, -809.7, -807.6, -812.9, -827.1]  # 右柱均值
# stds_left = [27.8, 38.8, 40.9, 44.8, 46.0, 38.2, 35.3]  # 右柱标准差
#
# # 可调整参数
# bar_width = 0.6  # 单根柱子的宽度
# group_spacing = 1.7  # 组间距离 (1.0是默认值，增大此值会增加组间距离)
#
# x = np.arange(len(groups)) * group_spacing  # 组的x位置，乘以组间距离系数
# width = bar_width  # 柱子的宽度
#
# fig, ax = plt.subplots(figsize=(12, 7))  # 进一步增加图形大小
#
#
# ########
# cmap = plt.get_cmap("coolwarm")
# # color1 = cmap(0.15)    # 最冷色（深蓝）
# # color2 = cmap(0.85)    # 最暖色（深红）

# # color1 = '#BEBAB9'    #
# # color2 = '#DBB8B2'    #
# color1 = '#C8D7EB'    #
# color2 = '#FAEBC7'    #
# # 绘制柱状图
# left_bars = ax.bar(x - width/2, means_left, width, yerr=stds_left, label='Lower', capsize=5, color = color1)
# right_bars = ax.bar(x + width/2, means_right, width, yerr=stds_right, label='Higher', capsize=5, color = color2)
#
# # 添加文本标签
# # def add_labels(bars, stds):
# #     for bar, std in zip(bars, stds):
# #         height = bar.get_height()
# #         ax.text(bar.get_x() + bar.get_width()/2., height + 1,
# #                 f'{height:.1f}±{std:.1f}',
# #                 ha='center', va='bottom', fontsize=15)
# #
# # add_labels(left_bars, stds_left)
# # add_labels(right_bars, stds_right)
#
# # # 添加文本标签
# def add_labels_left(bars, stds):
#     i=0
#     for bar, std in zip(bars, stds):
#         i=i+1
#         if((i==8)):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.135 + bar.get_width()/2., height+0.005,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif((i==8)):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.135 + bar.get_width()/2., height+0.007,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif(i==1):
#             height = bar.get_height()
#             ax.text(bar.get_x() -0.20 + bar.get_width()/2., height-2,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         else:
#             height = bar.get_height()
#             ax.text(bar.get_x() - 0.20 + bar.get_width() / 2., height,
#                     f'{height:.0f}±{std:.0f}',
#                     ha='center', va='bottom', fontsize=14)
# def add_labels_right(bars, stds):
#     i = 0
#     for bar, std in zip(bars, stds):
#         i=i+1
#         if(i==8):
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.1 + bar.get_width()/2., height-2.5,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         elif(i==2):
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.13 + bar.get_width()/2., height-1,
#                 f'{height:.0f}±{std:.0f}',
#                 ha='center', va='bottom', fontsize=14)
#         else:
#             height = bar.get_height()
#             ax.text(bar.get_x() + 0.13 + bar.get_width() / 2., height,
#                     f'{height:.0f}±{std:.0f}',
#                     ha='center', va='bottom', fontsize=14)
#
# add_labels_left(left_bars, stds_left)
# add_labels_right(right_bars, stds_right)
#
# # 添加组间比较的完整括号和星号
# def add_significance_brackets(ax, x1, x2, y, text, line_height=-860):
#     # 绘制完整的括号形状（水平线+两端竖线）
#     bracket_height = line_height
#     # 水平线
#     ax.plot([x1, x2], [y, y], lw=1, color='k')
#     # 左端竖线
#     ax.plot([x1, x1], [-870,-867], lw=1, color='k')
#     # 右端竖线
#     ax.plot([x2, x2], [-870,-867], lw=1, color='k')
#     # 添加星号文本（位置在水平线上方）
#     ax.text((x1+x2)*0.5, y + 1, text,
#             ha='center', va='bottom', fontsize=16)
#
# # 假设的显著性结果 (这里需要替换为你的实际统计结果)
# # 格式: (group_index, y_position, significance_text)
# significance_data = [
#     (0, -870, ''),    # 第1组显著
#     (1, -870, ''),    # 第2组显著
#     (2, -870, '**'),    # 第3组显著
#     (3, -870, ''),    # 第4组显著
#     (4, -870, ''),     # 第5组不显著
#     (5, -870, '**'),    # 第6组显著
#     (6, -870, ''),     # 第7组不显著
# ]
#
# for i, y, text in significance_data:
#     if((i!=0)&(i!=1)&(i!=3)&(i!=4)&(i!=6)):
#         x1 = x[i] - width/2 - 0.1
#         x2 = x[i] + width/2 + 0.1
#         add_significance_brackets(ax, x1, x2, y, text)
#
#
# plt.ylim(-780, -880)  # 调整y轴范围以容纳显著性标记
# plt.yticks(np.arange(-780, -880, -20))  # 30, 40, 50, 60, 70, 80
#
# ax.set_xticks(x)
# ax.set_xticklabels(groups)
# ax.set_ylabel('Lung density')
# ax.set_xlabel('Age group')
# ax.set_title('')
# ax.legend()
#
# plt.tight_layout()
# # plt.show()
# output_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/第一步/resnet34/resnet_34/肺密度.png'
# plt.savefig(output_path, bbox_inches='tight', dpi=600)





######拟合
# import numpy as np
# from scipy.optimize import curve_fit
#
# # 您的原始数据（频率Hz，测量峰峰值mV）
# frequency_data = np.array([1000, 5000, 10000, 20000, 30000, 40000, 50000,
#                           60000, 70000, 80000, 90000, 100000])
# measured_pp = np.array([1000, 1010, 1020, 1028, 1030, 1032, 1033.5,
#                         1034.5, 1035, 1036.5, 1036.6, 1036.7])
#
# # 计算校正因子（真实值/测量值）
# correction_factors = 1000 / measured_pp*1000000000
#
# # 定义三次多项式函数
# def cubic_poly(x, a, b, c, d):
#     return a*x**3 + b*x**2 + c*x + d
#
# # 拟合校正因子曲线
# popt, pcov = curve_fit(cubic_poly, frequency_data, correction_factors)
#
# # 获取拟合系数
# a, b, c, d = popt
# print(f"校正多项式系数: {a:.3e}, {b:.3e}, {c:.3e}, {d:.3e}")
#
# # 创建单一校正函数
# def universal_correction(freq, measured_value):
#     """适用于1kHz以上所有频率的通用校正公式"""
#     correction = cubic_poly(freq, *popt)
#     return measured_value * correction










# import os
# import shutil
# import openpyxl
#
#
# def copy_files_based_on_excel(excel_path, folder_a, folder_b):
#     """
#     根据Excel第一列的文件名，从folder_a复制文件到folder_b
#
#     参数:
#         excel_path: Excel文件路径
#         folder_a: 源文件夹路径
#         folder_b: 目标文件夹路径
#     """
#     # 确保目标文件夹存在
#     os.makedirs(folder_b, exist_ok=True)
#
#     # 加载Excel文件
#     wb = openpyxl.load_workbook(excel_path)
#     sheet = wb.active
#
#     # 遍历第一列的所有单元格（跳过可能的标题行）
#     for row in sheet.iter_rows(min_col=1, max_col=1, values_only=True):
#         filename = row[0]
#         if filename is None:  # 跳过空单元格
#             continue
#
#         # 构造源文件和目标文件路径
#         src_path = os.path.join(folder_a, str(filename))
#         dst_path = os.path.join(folder_b, str(filename))
#
#         # 检查文件是否存在并复制
#         if os.path.exists(src_path):
#             shutil.copy2(src_path, dst_path)
#             print(f"已复制: {filename}")
#         else:
#             print(f"文件不存在: {filename}")
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 替换为你的实际路径
#     excel_file = "/home/zsq/train/pre_process/DATE/再次做印证模型选择/test.xlsx"  # Excel文件路径
#     source_folder = "/home/zsq/train/pre_process/5378"  # 源文件夹路径
#     target_folder = "/home/zsq/train/pre_process/test_1"  # 目标文件夹路径
#
#     copy_files_based_on_excel(excel_file, source_folder, target_folder)







# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np # 仍然保留，因为绘图库需要它
#
# # 1. 强制读取 Excel 文件
# file_name = r'C:\Users\hp\Desktop\国庆工作\绘图\1.xlsx'
#
# # 使用 try-except 块来处理文件读取错误，但不再提供模拟数据
# try:
#     df = pd.read_excel(file_name)
#     # 假设第一列是种类，第二列是数字
#     category_column = df.columns[0]
#     value_column = df.columns[1]
# except FileNotFoundError:
#     # 如果文件未找到，直接终止程序并打印错误
#     raise FileNotFoundError(f"错误：未找到指定的 Excel 文件路径：{file_name}。请检查路径是否正确。")
# except IndexError:
#     # 如果文件只有一列，无法获取第二列（索引1）
#     raise ValueError("错误：Excel 文件中至少需要两列数据（种类和数值）。")
#
# # --- 关键检查步骤 ---
# print("--- 数据检查 ---")
# # 打印前几行数据，确认读取的内容是否正确
# print("DataFrame 前 5 行数据：")
# print(df.head())
# # 打印数值列的统计信息，确认数值范围
# print("\n数值列统计摘要 (Value Column Describe)：")
# print(df[value_column].describe())
# print("------------------")
#
# # 2. 创建图表
# plt.figure(figsize=(14, 7))
#
# # --- A. 绘制密度包络（小提琴图） ---
# sns.violinplot(
#     x=category_column,
#     y=value_column,
#     data=df,
#     inner=None,
#     palette="Pastel1",
#     linewidth=1.5,
#     saturation=0.5,
#     alpha=0.6
# )
#
# # --- B. 绘制点图（显示每个数据点）：建议使用 swarmplot 减少重叠，让点数更清晰 ---
# sns.swarmplot( # 将 stripplot 替换为 swarmplot
#     x=category_column,
#     y=value_column,
#     data=df,
#     color="black",
#     size=3,
#     linewidth=0.5 # 增加边框，让点在包络上更清晰
# )
#
# # 3. 添加标题和标签
# plt.title('Distribution of Value by Kind (Violin Plot + Swarm Plot)', fontsize=16)
# plt.xlabel(category_column, fontsize=12)
# plt.ylabel(value_column, fontsize=12)
#
# # 旋转 x 轴标签，防止重叠
# plt.xticks(rotation=45, ha='right')
#
# # 4. 显示图表
# plt.tight_layout()
# plt.show()