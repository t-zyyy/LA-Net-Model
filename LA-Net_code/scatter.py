# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# from scipy.stats import pearsonr
#
# # 选择系统已有的中文字体
# font_path = "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc"  # 文泉驿正黑
#
# # 加载字体
# font_prop = fm.FontProperties(fname=font_path)
#
# # 设置 Matplotlib 全局字体
# plt.rcParams['font.family'] = font_prop.get_name()
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
# # 读取 Excel 文件
# file_path = "/home/zsq/train/pre_process/DATE/COPD_new/正常组pian.xlsx"  # 替换为你的文件路径
# df = pd.read_excel(file_path)
#
# # 确保列名正确
# bias = df.iloc[:, 1]  # E列 偏差值
# FEV1 = df.iloc[:, 3]  # D列 FEV1 预测值
#
# # 初始化结果列表
# results = []
#
# # 遍历 group_size 从 1 到 100
# for group_size in range(1, 2):
#     # 按 group_size 进行分组计算均值
#     grouped_FEV1 = FEV1.groupby(FEV1.index // group_size).mean()
#     grouped_bias = bias.groupby(bias.index // group_size).mean()
#
#     # 计算相关系数（R值）、P值和R^2
#     if len(grouped_FEV1) > 1:  # 相关性计算至少需要两个点
#         corr, p_value = pearsonr(grouped_FEV1, grouped_bias)
#         r_squared = corr ** 2
#     else:
#         corr, p_value, r_squared = None, None, None
#
#     # 保存结果
#     results.append([group_size, corr, p_value, r_squared])
#
# # 创建 DataFrame 并保存到 Excel
# result_df = pd.DataFrame(results, columns=["group_size", "R", "P", "R^2"])
# output_path = "/home/zsq/train/pre_process/DATE/COPD_new/1.xlsx"
# result_df.to_excel(output_path, index=False)
#
# print(f"结果已保存到 {output_path}")









#######输出单个的偏差（3.25）
# import numpy as np
# from torch.utils.data import DataLoader
# from setting import parse_opts
# from brains import BrainS18Dataset
# from model import generate_model
# import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import os
# import torch
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# np.seterr(all='ignore')
#
#
# def run_ce(data_loader, model, img_names, sets, labels):
#     predictions = []
#     deviations = []
#     model.eval()
#
#     with tqdm(data_loader, desc="Processing Samples", unit="batch") as pbar:
#         for batch_id, batch_data in enumerate(pbar):
#             volume, _ = batch_data
#
#             if isinstance(volume, list):
#                 volume = torch.tensor(volume, dtype=torch.float32)
#             elif isinstance(volume, np.ndarray):
#                 volume = torch.tensor(volume, dtype=torch.float32)
#
#             if not isinstance(volume, torch.Tensor):
#                 raise TypeError(f"Expected volume to be a Tensor, but got {type(volume)}")
#
#             if not sets.no_cuda:
#                 volume = volume.cuda()
#
#             with torch.no_grad():
#                 pred = model(volume)
#                 pred = pred.cpu().numpy().flatten()
#                 predictions.append(pred[0])
#                 deviation = pred[0] - labels[batch_id]
#                 deviations.append(deviation)
#
#             tqdm.write(
#                 f"Batch {batch_id + 1}: Predicted Age = {pred[0]:.2f}, True Age = {labels[batch_id]:.2f}, Deviation = {deviation:.2f}")
#
#     return predictions, deviations
#
#
# if __name__ == '__main__':
#     sets = parse_opts()
#     sets.target_type = "age_regression"
#     sets.phase = 'test'
#     sets.test_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34(zengqiang)_5/resnet_34/epoch_87_batch_1_3.252322.pth.tar'
#     sets.input_D = 60
#     sets.test_root = '/home/zsq/train/COPD_60'
#     sets.test_file = '/home/zsq/train/pre_process/DATE/COPD_new/COPD.xlsx'
#
#     checkpoint = torch.load(sets.test_path, map_location=torch.device('cpu'))
#     model_state_dict = checkpoint['state_dict']
#     new_state_dict = {key.replace('module.', ''): value for key, value in model_state_dict.items()}
#
#     net, _ = generate_model(sets)
#     net.load_state_dict(new_state_dict)
#
#     n = 901
#     testing_data = BrainS18Dataset(sets.test_root, sets.test_file, sets)
#     subset_data = torch.utils.data.Subset(testing_data, range(n))
#     data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
#
#     df = pd.read_excel(sets.test_file)
#     img_names = df.iloc[:, 0].tolist()
#     labels = df.iloc[:, 1].tolist()[:n]
#
#     predictions, deviations = run_ce(data_loader, net, img_names, sets, labels)
#
#     output_df = pd.DataFrame({'Filename': img_names[:n], 'Deviation': deviations})
#     output_df.to_excel('/home/zsq/train/pre_process/DATE/COPD_new/output.xlsx', index=False)
#
#     print("Results saved to output.xlsx")




#
# ########研究比例
# import numpy as np
# import pandas as pd
#
# # -------------------------------
# # 1. 配置输入和输出文件路径
# # -------------------------------
# input_excel_path = '/home/zsq/train/pre_process/DATE/COPD_new/分析偏大偏小/三+四.xlsx'    # 你的偏差数据 Excel 文件路径
# output_excel_path = '/home/zsq/train/pre_process/DATE/COPD_new/分析偏大偏小/重-极重——偏大.xlsx'  # 结果输出文件路径
#
# # -------------------------------
# # 2. 从 Excel 中读取偏差数据
# # -------------------------------
# try:
#     df = pd.read_excel(input_excel_path)
# except Exception as e:
#     print("读取 Excel 文件失败：", e)
#     exit(1)
#
# # 假设偏差数据存储在名为 'C' 的列中；如果没有该列，则默认使用第一列数据
# if 'C' in df.columns:
#     biases = df['C'].values
# else:
#     biases = df.iloc[:, 2].values
#
# total_data = len(biases)
# print(f"总数据个数：{total_data}")
#
# # -------------------------------
# # 3. 计算各个阈值下低于该阈值的数据比例
# # -------------------------------
# # 这里设置阈值范围为 -5 到 -1（步长 0.1）
# thresholds = np.arange(0, 3.1, 0.1)  # 注意 5.1 确保包含 5
# proportions = []
#
# for thresh in thresholds:
#     # 统计偏差值小于当前阈值的数据个数
#     count = np.sum(biases > thresh)
#     # 计算比例，乘以100转换为百分比
#     ratio = (count / total_data) * 100
#     proportions.append(ratio)
#     # 如果需要调试，可以打印当前阈值和比例
#     # print(f"Threshold: {thresh:.1f}, Count: {count}, Proportion: {ratio:.2f}%")
#
# # -------------------------------
# # 4. 将结果保存到 Excel
# # -------------------------------
# # 构造 DataFrame，其中 A 列为阈值，B 列为对应的比例
# result_df = pd.DataFrame({
#     'Threshold': thresholds,
#     'Proportion (%)': proportions
# })
#
# # 保存到 Excel 文件中
# try:
#     result_df.to_excel(output_excel_path, index=False)
#     print(f"计算结果已保存到 {output_excel_path}")
# except Exception as e:
#     print("写入 Excel 文件失败：", e)











#########方差相等的t检验
# import numpy as np
# from scipy.stats import t
#
# # 定义参数
# mean1, std1, n1 = 47.39, 2.33, 49
# mean2, std2, n2 = 47.42, 90.04, 48
#
# # 计算合并标准差（假设方差相等）
# pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
#
# # 计算标准误差
# std_error = pooled_std * np.sqrt(1 / n1 + 1 / n2)
#
# # 计算 t 统计量
# t_stat = (mean1 - mean2) / std_error
#
# # 计算自由度
# df = n1 + n2 - 2
#
# # 计算双尾 p 值
# p_value = 2 * (1 - t.cdf(abs(t_stat), df))
#
# # 输出结果
# print(f"t统计量 = {t_stat:.3f}")
# print(f"自由度 = {df}")
# print(f"p值 = {p_value:.3f}")
# if p_value < 0.05:
#     print("→ 显著差异（拒绝原假设）")
# else:
#     print("→ 无显著差异（无法拒绝原假设）")








# ##########方差不等的t检验
# import numpy as np
# from scipy.stats import t
#
# # 定义参数
# mean1, std1, n1 = 67.3, 19.55, 41
# mean2, std2, n2 = 72.48, 16.95, 33
#
# # 计算 Welch's t 统计量
# t_stat = (mean1 - mean2) / np.sqrt(std1**2 / n1 + std2**2 / n2)
#
# # 计算 Welch's t 检验的自由度
# df = ((std1**2 / n1 + std2**2 / n2)**2) / ((std1**2 / n1)**2 / (n1 - 1) + (std2**2 / n2)**2 / (n2 - 1))
#
# # 计算双尾 p 值
# p_value = 2 * (1 - t.cdf(abs(t_stat), df))
#
# # 输出结果
# print(f"Welch's t统计量 = {t_stat:.3f}")
# print(f"自由度 = {df:.3f}")
# print(f"p值 = {p_value:.3f}")
# if p_value < 0.05:
#     print("→ 显著差异（拒绝原假设）")
# else:
#     print("→ 无显著差异（无法拒绝原假设）")









# ##############真正的双样本t检验
# import pandas as pd
# from scipy import stats
#
# # 指定 Excel 文件路径
# file_path = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/新的Pred绘图/新新的抽取/测.xlsx"  # 替换为你的 Excel 文件路径
#
# # 读取 Excel 文件
# try:
#     # 假设数据在第一个 sheet 中
#     df = pd.read_excel(file_path)
#
#     # 检查是否有至少两列数据
#     if df.shape[1] < 2:
#         print("❌ Excel 文件至少需要两列数据！")
#     else:
#         # 提取第一列和第二列数据
#         group1 = df.iloc[:, 0].dropna()  # 第一列数据，并去除缺失值
#         group2 = df.iloc[:, 1].dropna()  # 第二列数据，并去除缺失值
#
#         # 打印数据基本信息
#         print("第一组数据（列1）：")
#         print(group1.describe())
#         print("\n第二组数据（列2）：")
#         print(group2.describe())
#
#         # 进行双样本 t 检验
#         t_statistic, p_value = stats.ttest_ind(group1, group2)
#
#         # 输出 t 检验结果
#         print("\n双样本 t 检验结果：")
#         print(f"t 值: {t_statistic:.4f}")
#         print(f"p 值: {p_value:.4f}")
#
#         # 判断显著性
#         alpha = 0.05  # 显著性水平
#         if p_value < alpha:
#             print("结果显著：两组数据有显著差异（p < 0.05）")
#         else:
#             print("结果不显著：两组数据无显著差异（p >= 0.05）")
#
# except FileNotFoundError:
#     print(f"❌ 文件 {file_path} 未找到，请检查路径是否正确！")
# except Exception as e:
#     print(f"❌ 读取文件时出错：{e}")









#
# #########蝴蝶图
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator, FormatStrFormatter
#
# # 左侧（Group A）数据
# mean_left = [77.21, 75.28, 75.94, 73.87, 74.63, 74.37, 73.92, 72.48]
# std_left = [10.14, 12.60, 13.88, 12.57, 24.77, 15.16, 15.66, 16.48]
#
# # 右侧（Group B）数据
# mean_right = [74.52, 74.48, 73.13, 72.92, 73.06, 73.11, 71.74, 67.3]
# std_right = [15.95, 13.62, 13.99, 15.85, 14.97, 16.89, 18.53, 19.55]
#
# # 定义y轴的标签
# labels = ['35-43', '44-48', '49-51', '52-55', '56-57', '58-60', '61-65', '66-74']
#
# # 每个标签对应的星号数量（可以根据需要自定义）
# stars = [1, 1, 1, 1, 1, 1, 1, 1]  # 对应每个y轴标签的星号数量
#
# # 生成图形
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # 左侧柱状图（取负值实现镜像）
# ax.barh(labels, -np.array(mean_left), color='skyblue', label='Group A', alpha=0.7)
# # 右侧柱状图
# ax.barh(labels, mean_right, color='lightcoral', label='Group B', alpha=0.7)
#
# # 标注均值 ± 方差，并添加"T"形线段
# for i, (ml, sl, mr, sr) in enumerate(zip(mean_left, std_left, mean_right, std_right)):
#     # 左侧标注（贴着x轴的0.0左侧）
#     ax.text(-2, labels[i], f'{sl:.2f} ± {ml:.2f}', color='blue', va='center', ha='right')
#
#     # 右侧标注（贴着x轴的0.0右侧）
#     ax.text(2, labels[i], f'{mr:.2f} ± {sr:.2f}', color='red', va='center', ha='left')
#
#     # 左侧"T"形线段
#     ax.errorbar(-ml, labels[i], xerr=sl, fmt='none', color='blue', capsize=5, capthick=1.5)
#     # 右侧"T"形线段
#     ax.errorbar(mr, labels[i], xerr=sr, fmt='none', color='red', capsize=5, capthick=1.5)
#
#     # 在y轴标签左上角添加星号（从stars数组中获取每个标签对应的星号数量）
#     star_count = stars[i]  # 获取当前标签需要的星号数量
#     star = '*' * star_count  # 生成对应数量的星号
#     ax.text(-122.5, labels[i], f' {star}', color='black', va='center', ha='right', fontsize=12)
#
# # 设置x轴标签格式，防止重叠
# max_value = max(max(mean_left), max(mean_right))
# ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # 控制标签数量
# ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # 将x轴标签转换为整数（去掉小数）
#
# # 添加分隔线和图例
# ax.axvline(0, color='black', lw=1)
# ax.set_xlabel('FEV1/FEV1_Pred')
# ax.set_ylabel('Age Group', labelpad=20)  # 增加 labelpad 的值
#
#
# # 添加图例
# # ax.legend()
#
# # 美化外观
# plt.title('')
# plt.grid(True, linestyle='--', alpha=0.5)
#
# # 旋转标签（防止重叠）
# plt.xticks(rotation=0)
#
# # 显示图形
# plt.show()
#








# #########蝴蝶图---每个年龄的
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator, FormatStrFormatter
#
# # 左侧（Group A）数据
# mean_left = [79.27, 75.13, 76.52, 88.51, 74.01, 78.61, 76.53, 74.73,
#              75.09,67.3,71.74,79.06,70.08,79.82,75.77,80.12,76.86,70.25,
#              67.98,82.80,70.61,78.22,72.06,74.13,69.93,76.02,75.23,77.7,
#              73.40,56.43,80.70,80.70,70.73]
# std_left = [3.81, 9.69,9.07, 1.70,15.40,13.32, 16.20,13.43, 5.77, 17.67, 19.61,10.19,19.06,
#             6.94,10.15, 10.04,14.01, 11.75, 21.06, 19.74, 21.84,15.74,16.38,
#             13.35, 13.33, 6.76,15.84,10.45,15.76, 15.68,18.43, 10.83, 22.80]
#
# # 右侧（Group B）数据
# mean_right = [74.96, 83.86, 78.39, 75.52, 88.30, 72.30, 65.60, 72.72,
#               71.23,72.93,79.05,70.36,68.12,70.66,72.97,74.39,74.47,74.62,
#               81.35,80.71,68.01,79.02,79.52,71.05,88.37,51.36, 71.40,65.,
#               73.40,43.88,74.45,74.45,70.71]
# std_right = [17.16,  13.88,10.40,  7.48,10.0,13.82, 12.38,21.22, 15.93, 13.35, 6.51, 18.57,19.93,
#              6.96,13.23, 14.68,10.46,  23.69, 10.10,14.08,  18.65,13.96,  14.08,
#              23.08, 11.79, 16.28, 16.81,11.56, 19.50,  5.3,11.71,  14.36,  16.40
#              ]
#
# # 定义y轴的标签
# labels = ['35-36', '37**', '38-39**', '40', '41**', '42', '43', '44', '45', '46**', '47**', '48',
#           '49', '50', '51', '52', '53', '54**', '55**', '56', '57', '58**', '59**', '60', '61**',
#           '62', '63', '64', '65', '66', '67', '68', '69-74']
#
# # 每个标签对应的星号数量（可以根据需要自定义）
# stars = [0, 0, 0, 0, 0, 0, 0, 0]  # 对应每个y轴标签的星号数量
#
# # 生成图形
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # 左侧柱状图（取负值实现镜像）
# ax.barh(labels, -np.array(mean_left), color='skyblue', label='Group A', alpha=0.7)
# # 右侧柱状图
# ax.barh(labels, mean_right, color='lightcoral', label='Group B', alpha=0.7)
#
# # 标注均值 ± 方差，并添加"T"形线段
# for i, (ml, sl, mr, sr) in enumerate(zip(mean_left, std_left, mean_right, std_right)):
#     # 左侧标注（贴着x轴的0.0左侧）
#     ax.text(-2, labels[i], f'{sl:.2f} ± {ml:.2f}', color='blue', va='center', ha='right')
#
#     # 右侧标注（贴着x轴的0.0右侧）
#     ax.text(2, labels[i], f'{mr:.2f} ± {sr:.2f}', color='red', va='center', ha='left')
#
#     # 左侧"T"形线段
#     ax.errorbar(-ml, labels[i], xerr=sl, fmt='none', color='blue', capsize=5, capthick=1.5)
#     # 右侧"T"形线段
#     ax.errorbar(mr, labels[i], xerr=sr, fmt='none', color='red', capsize=5, capthick=1.5)
#
#     # 在y轴标签左上角添加星号（从stars数组中获取每个标签对应的星号数量）
#     star_count = 0#stars[i]  # 获取当前标签需要的星号数量
#     star = '*' * star_count  # 生成对应数量的星号
#     ax.text(-122.5, labels[i], f' {star}', color='black', va='center', ha='right', fontsize=12)
#
# # 设置x轴标签格式，防止重叠
# max_value = max(max(mean_left), max(mean_right))
# ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # 控制标签数量
# ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # 将x轴标签转换为整数（去掉小数）
#
# # 添加分隔线和图例
# ax.axvline(0, color='black', lw=1)
# ax.set_xlabel('FEV1/FEV1_Pred')
# ax.set_ylabel('Age Group', labelpad=20)  # 增加 labelpad 的值
#
#
# # 添加图例
# # ax.legend()
#
# # 美化外观
# plt.title('')
# plt.grid(True, linestyle='--', alpha=0.5)
#
# # 旋转标签（防止重叠）
# plt.xticks(rotation=0)
#
# # 显示图形
# plt.show()







# import pandas as pd
# from scipy.stats import mannwhitneyu, ks_2samp, brunnermunzel, wilcoxon
# import numpy as np
# import warnings
#
# # 忽略警告
# warnings.filterwarnings("ignore")
#
#
# # ========================= 读取数据（修复版本） =========================
# def read_data(file_path):
#     """
#     强化数据类型转换和异常处理
#     """
#     df = pd.read_excel(file_path)
#
#     # # 双重保险转换机制
#     # def safe_convert(column):
#     #     # 先尝试整体转换为浮点型
#     #     converted = pd.to_numeric(column, errors='coerce')
#     #     # 二次检查：逐个元素转换，处理特殊字符
#     #     cleaned = []
#     #     for item in converted:
#     #         try:
#     #             cleaned.append(float(item))
#     #         except:
#     #             cleaned.append(np.nan)
#     #     return pd.Series(cleaned).dropna().values
#
#     group1 = df.iloc[:, 9].dropna()  # 第一列数据，并去除缺失值
#     group2 = df.iloc[:, 10].dropna()  # 第二列数据，并去除缺失值
#
#     #         # 提取第一列和第二列数据
#
#     # 空数据校验
#     if len(group1) < 3 or len(group2) < 3:  # 至少需要3个数据点
#         raise ValueError("有效数据量不足（需≥3），请检查数据质量")
#
#     # 强制类型声明
#     return np.asarray(group1, dtype=np.float64), np.asarray(group2, dtype=np.float64)
#
#
# # ========================= 执行检验 =========================
# def perform_tests(group1, group2, alpha=0.05):
#     results = {}
#
#     # # 1. Mann-Whitney U检验
#     # try:
#     #     stat, p = mannwhitneyu(group1, group2,alternative='two-sided')
#     #     results["Mann-Whitney U"] = {
#     #         "stat": str(stat),
#     #         "p": str(p),
#     #     }
#     # except Exception as e:
#     #     results["Mann-Whitney U"] = {"error": f"检验失败: {str(e)}"}
#
#     # # 2. Kolmogorov-Smirnov检验
#     # try:
#     #     stat, p = ks_2samp(group1, group2)
#     #     results["Kolmogorov-Smirnov"] = {
#     #         "stat": stat,
#     #         "p": p,
#     #         "significant": p < alpha
#     #     }
#     # except Exception as e:
#     #     results["Kolmogorov-Smirnov"] = {"error": f"检验失败: {str(e)}"}
#     #
#     # # 3. Brunner-Munzel检验
#     # try:
#     #     stat, p = brunnermunzel(group1, group2)
#     #     results["Brunner-Munzel"] = {
#     #         "stat": stat,
#     #         "p": p,
#     #         "significant": p < alpha
#     #     }
#     # except Exception as e:
#     #     results["Brunner-Munzel"] = {"error": f"检验失败: {str(e)}"}
#
#     # 4. Wilcoxon符号秩检验（仅当数据长度相同时执行）
#     # try:
#     #     if len(group1) == len(group2):
#     #         stat, p = wilcoxon(group1, group2)
#     #         results["Wilcoxon符号秩"] = {
#     #             "stat": stat,
#     #             "p": p,
#     #             "significant": p < alpha
#     #         }
#     #     else:
#     #         results["Wilcoxon符号秩"] = {"error": "数据长度不同，无法执行配对检验"}
#     # except Exception as e:
#     #     results["Wilcoxon符号秩"] = {"error": f"检验失败: {str(e)}"}
#
#     return results
#
#
# # ========================= 结果输出 =========================
# def print_results(results):
#     for test_name, result in results.items():
#         print(f"\n=== {test_name}检验 ===")
#         if 'error' in result:
#             print(f"错误信息: {result['error']}")
#         else:
#             print(f"统计量: {result['stat']:.10f}")
#             print(f"p值: {result['p']:.20f}")
#             print(f"是否显著 (α=0.05): {'是' if result['significant'] else '否'}")
#
#
# # ========================= 主程序 =========================
# if __name__ == "__main__":
#     file_path = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/重来吧/之前的/57-60.xlsx"  # 修改为你的Excel路径
#
#     try:
#         group1, group2 = read_data(file_path)
#         print(f"读取成功 | 组1样本量: {len(group1)}, 组2样本量: {len(group2)}")
#         print(f"组1示例数据: {group1[:5]}...")  # 打印前5个数据点供检查
#         print(f"组2示例数据: {group2[:5]}...")
#     except Exception as e:
#         print(f"错误: {str(e)}")
#         exit()
#
#     results = perform_tests(group1, group2)
#     print("\n============= 检验结果 =============")
#     print_results(results)













import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import brunnermunzel
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
def load_excel_data(file_path):
    """
    从Excel文件加载两组数据
    :param file_path: Excel文件路径
    :return: 两组数据 (group1, group2)
    """
    df = pd.read_excel(file_path, header=None, engine='openpyxl')

    # 提取前两列，并自动处理缺失值和类型转换
    group1 = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
    group2 = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values

    # 数据检查
    if len(group1) < 3 or len(group2) < 3:
        raise ValueError("每组至少需要3个有效数据点")

    print(f"数据加载成功: 组1有{len(group1)}个样本，组2有{len(group2)}个样本")
    return group1, group2

#
# def perform_mannwhitney(group1, group2):
#     """
#     执行Mann-Whitney U检验并返回完整结果
#     """
#     # 执行检验
#     stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
#
#     # 计算效应量 (Cliff's Delta)
#     def cliffs_delta(x, y):
#         n_x, n_y = len(x), len(y)
#         pairs = [(a, b) for a in x for b in y]
#         wins = sum(1 for (a, b) in pairs if a > b)
#         losses = sum(1 for (a, b) in pairs if a < b)
#         return (wins - losses) / (n_x * n_y)
#
#     d = cliffs_delta(group1, group2)
#
#     # 计算中位数
#     median1, median2 = np.median(group1), np.median(group2)
#
#     return {
#         'U_statistic': stat,
#         'p_value': p,
#         'cliffs_delta': d,
#         'group1_median': median1,
#         'group2_median': median2,
#         'interpretation': '显著差异 (p < 0.05)' if p < 0.05 else '无显著差异'
#     }
#
#
# def perform_ks_test(group1, group2):
#     """
#     执行Kolmogorov-Smirnov检验并返回完整结果
#     （保持与Mann-Whitney检验相同的输出格式）
#
#     参数:
#         group1, group2: 两组待比较的数据 (array-like)
#
#     返回:
#         {
#             'statistic': KS统计量D,
#             'p_value': p值,
#             'effect_size': 效应量,
#             'group1_ecdf': 组1的经验分布函数值,
#             'group2_ecdf': 组2的经验分布函数值,
#             'interpretation': 结果解读
#         }
#     """
#     # 执行KS检验
#     stat, p = ks_2samp(group1, group2)
#
#     # 计算效应量（KS统计量本身就是效应量）
#     effect_size = stat
#
#     # 计算中位数（保持格式一致）
#     median1, median2 = np.median(group1), np.median(group2)
#
#     # 计算经验分布函数（用于可视化）
#     def ecdf(x):
#         xs = np.sort(x)
#         ys = np.arange(1, len(xs) + 1) / float(len(xs))
#         return xs, ys
#
#     ecdf1 = ecdf(group1)
#     ecdf2 = ecdf(group2)
#
#     return {
#         'statistic': stat,
#         'p_value': p,
#         'effect_size': effect_size,
#         'group1_median': median1,
#         'group2_median': median2,
#         'group1_ecdf': ecdf1,
#         'group2_ecdf': ecdf2,
#         'interpretation': '分布显著不同 (p < 0.05)' if p < 0.05 else '分布无显著差异'
#     }
#
#
# def perform_brunnermunzel_test(group1, group2):
#     """
#     执行Brunner-Munzel检验并返回完整结果
#     （保持与Mann-Whitney/KS检验相同的输出格式）
#
#     参数:
#         group1, group2: 两组待比较的数据 (array-like)
#
#     返回:
#         {
#             'statistic': BM统计量,
#             'p_value': p值,
#             'effect_size': 效应量 (P(X<Y) - P(X>Y)),
#             'group1_median': 组1中位数,
#             'group2_median': 组2中位数,
#             'interpretation': 结果解读
#         }
#     """
#     # 执行Brunner-Munzel检验
#     try:
#         stat, p = brunnermunzel(group1, group2)
#
#         # 计算效应量 (P(X<Y) - P(X>Y))
#         def calculate_effect_size(x, y):
#             n_x, n_y = len(x), len(y)
#             pairs = [(a, b) for a in x for b in y]
#             p_xy = sum(1 for (a, b) in pairs if a < b) / (n_x * n_y)
#             p_yx = sum(1 for (a, b) in pairs if a > b) / (n_x * n_y)
#             return p_xy - p_yx
#
#         effect_size = calculate_effect_size(group1, group2)
#
#         # 计算中位数
#         median1, median2 = np.median(group1), np.median(group2)
#
#         return {
#             'statistic': stat,
#             'p_value': p,
#             'effect_size': effect_size,
#             'group1_median': median1,
#             'group2_median': median2,
#             'interpretation': '显著差异 (p < 0.05)' if p < 0.05 else '无显著差异'
#         }
#
#     except Exception as e:
#         return {
#             'error': f"检验失败: {str(e)}",
#             'interpretation': '无法执行检验'
#         }
#
#
#
#
# def perform_wilcoxon_test(group1, group2):
#     """
#     执行配对Wilcoxon符号秩检验
#
#     参数:
#         group1: 第一组数据 (如治疗前)
#         group2: 第二组数据 (如治疗后)
#
#     返回:
#         {
#             'statistic': 统计量,
#             'p_value': p值,
#             'effect_size': 效应量 (r = Z/√n),
#             'median_diff': 中位数差值 (group2 - group1),
#             'interpretation': 结果解读,
#             'descriptive_stats': 描述性统计
#         }
#     """
#     # 检查数据长度是否匹配
#     if len(group1) != len(group2):
#         return {
#             'error': f"数据长度不匹配 (group1:{len(group1)}, group2:{len(group2)})",
#             'interpretation': '无法执行配对检验'
#         }
#
#     # 执行检验
#     try:
#         stat, p = wilcoxon(group2, group1)  # 注意顺序：计算group2 - group1
#
#         # 计算效应量 (r = Z/√n)
#         n = len(group1)
#         r = abs(stat / np.sqrt(n)) if n > 0 else 0
#
#         # 计算描述性统计
#         median_diff = np.median(np.array(group2) - np.array(group1))
#         mean_diff = np.mean(np.array(group2) - np.mean(np.array(group1)))
#
#         return {
#             'statistic': stat,
#             'p_value': p,
#             'effect_size': r,
#             'median_diff': median_diff,
#             'mean_diff': mean_diff,
#             'interpretation': '差异显著 (p < 0.05)' if p < 0.05 else '差异不显著',
#             'descriptive_stats': {
#                 'group1_median': np.median(group1),
#                 'group2_median': np.median(group2),
#                 'n_pairs': n
#             }
#         }
#     except Exception as e:
#         return {
#             'error': f"检验失败: {str(e)}",
#             'interpretation': '无法执行检验'
#         }
#
#
#
# from itertools import product
#
#
# def cliffs_delta(x, y, verbose=False):
#     """
#     计算两组数据的Cliff's Delta效应量
#
#     参数:
#         x, y: 两组独立数据 (array-like)
#         verbose: 是否打印详细计算过程
#
#     返回:
#         {
#             'delta': 效应量 (-1 ≤ δ ≤ 1),
#             'magnitude': 效应大小描述,
#             'interpretation': 统计解释,
#             'comparison': 优势比较,
#             'overlap': 重叠比例
#         }
#     """
#     # 数据预处理
#     x, y = np.asarray(x), np.asarray(y)
#     n_x, n_y = len(x), len(y)
#
#     # 计算优势对 (dominance counts)
#     pairs = list(product(x, y))
#     wins = sum(1 for (a, b) in pairs if a > b)
#     losses = sum(1 for (a, b) in pairs if a < b)
#     ties = len(pairs) - wins - losses
#
#     # 计算Cliff's Delta
#     delta = (wins - losses) / (n_x * n_y)
#
#     # 计算重叠比例
#     overlap = (1 - abs(delta)) / 2
#
#     # 效应大小分类 (Romano et al. 2006)
#     magnitude = ""
#     if abs(delta) < 0.147:
#         magnitude = "微不足道"
#     elif abs(delta) < 0.33:
#         magnitude = "小效应"
#     elif abs(delta) < 0.474:
#         magnitude = "中等效应"
#     else:
#         magnitude = "大效应"
#
#     # 优势方向解释
#     comparison = ""
#     if delta > 0:
#         comparison = f"组1有 {delta:.0%} 概率大于组2"
#     elif delta < 0:
#         comparison = f"组2有 {abs(delta):.0%} 概率大于组1"
#     else:
#         comparison = "两组无优势差异"
#
#     # 详细输出
#     if verbose:
#         print(f"\nCliff's Delta 计算详情:")
#         print(f"  - 组1样本量: {n_x}")
#         print(f"  - 组2样本量: {n_y}")
#         print(f"  - 优势对 (组1>组2): {wins}")
#         print(f"  - 劣势对 (组1<组2): {losses}")
#         print(f"  - 平局对: {ties}")
#         print(f"  - 总比较对数: {n_x * n_y}")
#
#     return {
#         'delta': float(delta),
#         'magnitude': magnitude,
#         'interpretation': f"效应量 δ = {delta:.3f} ({magnitude})",
#         'comparison': comparison,
#         'overlap': float(overlap),
#         'stats': {
#             'wins': wins,
#             'losses': losses,
#             'ties': ties,
#             'n_x': n_x,
#             'n_y': n_y
#         }
#     }
#
# from scipy.stats import chi2_contingency, median_test
#
#
# def moods_median_test_two_groups(group1, group2, alpha=0.05):
#     """
#     两组数据的Mood's中位数检验
#
#     参数:
#         group1, group2: 两组独立数据 (array-like)
#         alpha: 显著性水平
#
#     返回:
#         {
#             'statistic': 卡方统计量,
#             'p_value': p值,
#             'grand_median': 合并中位数,
#             'group1_median': 组1中位数,
#             'group2_median': 组2中位数,
#             'interpretation': 结果解释,
#             'contingency_table': 列联表
#         }
#     """
#     # 数据检查
#     group1, group2 = np.asarray(group1), np.asarray(group2)
#     if len(group1) < 5 or len(group2) < 5:
#         print("警告：样本量小于5，检验效能可能不足")
#
#     # 执行检验
#     stat, p, med, tbl = median_test(group1, group2)
#
#     # 结果解释
#     interpretation = (
#         f"拒绝原假设 (p={p:.4f})，两组中位数不同"
#         if p < alpha
#         else f"不拒绝原假设 (p={p:.4f})"
#     )
#
#     return {
#         'statistic': stat,
#         'p_value': p,
#         'grand_median': med,
#         'group1_median': np.median(group1),
#         'group2_median': np.median(group2),
#         'interpretation': interpretation,
#         'contingency_table': tbl
#     }
#
#
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import numpy as np


# def bootstrap_ttest(group1, group2, n_bootstrap=9999, alternative='two-sided', ci_level=0.95, random_seed=None):
#     """
#     Bootstrap t检验
#
#     参数:
#         group1, group2: 两组独立数据 (array-like)
#         n_bootstrap: 重抽样次数 (默认9999)
#         alternative: 检验类型 ('two-sided', 'less', 'greater')
#         ci_level: 置信区间水平 (默认0.95)
#         random_seed: 随机种子
#
#     返回:
#         {
#             't_original': 原始t统计量,
#             'p_value': p值,
#             'ci_lower': 置信区间下限,
#             'ci_upper': 置信区间上限,
#             'mean_diff': 均值差异 (group1 - group2),
#             'bootstrap_dist': Bootstrap t统计量分布,
#             'interpretation': 结果解释
#         }
#     """
#     if random_seed is not None:
#         np.random.seed(random_seed)
#
#     group1, group2 = np.asarray(group1), np.asarray(group2)
#     n1, n2 = len(group1), len(group2)
#
#     # 计算原始 t 统计量
#     mean_diff = np.mean(group1) - np.mean(group2)
#     pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
#     se = pooled_std * np.sqrt(1 / n1 + 1 / n2)
#     t_original = mean_diff / se
#
#     # Bootstrap 过程
#     combined = np.concatenate([group1, group2])
#     t_bootstrap = np.zeros(n_bootstrap)
#
#     for i in range(n_bootstrap):
#         # 有放回重抽样
#         sample1 = np.random.choice(combined, size=n1, replace=True)
#         sample2 = np.random.choice(combined, size=n2, replace=True)
#
#         # 计算 Bootstrap t 统计量
#         diff = np.mean(sample1) - np.mean(sample2)
#         pooled_std_bs = np.sqrt(
#             ((n1 - 1) * np.var(sample1, ddof=1) + (n2 - 1) * np.var(sample2, ddof=1)) / (n1 + n2 - 2))
#         se_bs = pooled_std_bs * np.sqrt(1 / n1 + 1 / n2)
#         t_bootstrap[i] = diff / se_bs
#
#     # 计算 p 值
#     if alternative == 'two-sided':
#         p_value = np.sum(np.abs(t_bootstrap) >= np.abs(t_original)) / n_bootstrap
#     elif alternative == 'less':
#         p_value = np.sum(t_bootstrap <= t_original) / n_bootstrap
#     elif alternative == 'greater':
#         p_value = np.sum(t_bootstrap >= t_original) / n_bootstrap
#     else:
#         raise ValueError("alternative 必须是 'two-sided', 'less' 或 'greater'")
#
#     # 计算置信区间
#     ci_lower = np.percentile(t_bootstrap, (1 - ci_level) / 2 * 100)
#     ci_upper = np.percentile(t_bootstrap, (1 + ci_level) / 2 * 100)
#
#     # 结果解释
#     alpha = 1 - ci_level
#     if p_value < alpha:
#         if alternative == 'two-sided':
#             interp = f"拒绝原假设 (p={p_value:.4f})，两组均值存在显著差异"
#         elif alternative == 'less':
#             interp = f"拒绝原假设 (p={p_value:.4f})，组1均值显著小于组2"
#         else:
#             interp = f"拒绝原假设 (p={p_value:.4f})，组1均值显著大于组2"
#     else:
#         interp = f"不拒绝原假设 (p={p_value:.4f})"
#
#     return {
#         't_original': t_original,
#         'p_value': p_value,
#         'ci_lower': ci_lower,
#         'ci_upper': ci_upper,
#         'mean_diff': mean_diff,
#         'bootstrap_dist': t_bootstrap,
#         'interpretation': interp,
#         'n_bootstrap': n_bootstrap,
#         'alternative': alternative
#     }


def bootstrap_ttest(group1, group2, n_bootstrap=9999, alternative='two-sided', ci_level=0.95, random_seed=None):
    """
    Bootstrap t检验

    参数:
        group1, group2: 两组独立数据 (array-like)
        n_bootstrap: 重抽样次数 (默认9999)
        alternative: 检验类型 ('two-sided', 'less', 'greater')
        ci_level: 置信区间水平 (默认0.95)
        random_seed: 随机种子

    返回:
        {
            't_original': 原始t统计量,
            'p_value': p值,
            'ci_lower': 置信区间下限,
            'ci_upper': 置信区间上限,
            'mean_diff': 均值差异 (group1 - group2),
            'bootstrap_dist': Bootstrap t统计量分布,
            'interpretation': 结果解释
        }
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    group1, group2 = np.asarray(group1), np.asarray(group2)
    n1, n2 = len(group1), len(group2)

    # 计算原始 t 统计量
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    t_original = mean_diff / (pooled_std * np.sqrt(1/n1 + 1/n2))

    # Bootstrap 过程
    combined = np.concatenate([group1, group2])
    t_bootstrap = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample1 = np.random.choice(combined, size=n1, replace=True)
        sample2 = np.random.choice(combined, size=n2, replace=True)

        # 计算 Bootstrap t 统计量
        diff = np.mean(sample1) - np.mean(sample2)
        pooled_std_bs = np.sqrt(((n1 - 1) * np.var(sample1, ddof=1) + (n2 - 1) * np.var(sample2, ddof=1)) / (n1 + n2 - 2))
        t_bootstrap[i] = diff / (pooled_std_bs * np.sqrt(1/n1 + 1/n2))

    # 计算 p 值
    if alternative == 'two-sided':
        p_value = np.sum(np.abs(t_bootstrap) >= np.abs(t_original)) / n_bootstrap
    elif alternative == 'less':
        p_value = np.sum(t_bootstrap <= t_original) / n_bootstrap
    elif alternative == 'greater':
        p_value = np.sum(t_bootstrap >= t_original) / n_bootstrap
    else:
        raise ValueError("alternative 必须是 'two-sided', 'less' 或 'greater'")

    # 计算置信区间
    ci_lower = np.percentile(t_bootstrap, (1 - ci_level) / 2 * 100)
    ci_upper = np.percentile(t_bootstrap, (1 - (1 - ci_level) / 2) * 100)

    # 结果解释
    alpha = 1 - ci_level
    if p_value < alpha:
        if alternative == 'two-sided':
            interp = f"拒绝原假设 (p={p_value:.4f})，两组均值存在显著差异"
        elif alternative == 'less':
            interp = f"拒绝原假设 (p={p_value:.4f})，组1均值显著小于组2"
        else:
            interp = f"拒绝原假设 (p={p_value:.4f})，组1均值显著大于组2"
    else:
        interp = f"不拒绝原假设 (p={p_value:.4f})"

    return {
        't_original': t_original,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_diff': mean_diff,
        'bootstrap_dist': t_bootstrap,
        'interpretation': interp,
        'n_bootstrap': n_bootstrap,
        'alternative': alternative
    }






if __name__ == "__main__":
    # 1. 加载数据
    excel_path = "/home/zsq/train/pre_process/DATE/外部验证excel/其余关系/算均值±方差/11.xlsx"  # 替换为你的Excel文件路径
    try:
        group1, group2 = load_excel_data(excel_path)
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit()



# #
#     # 2. 执行检验
#     result1 = perform_mannwhitney(group1, group2)
#     ks_result =perform_ks_test(group1, group2)
#     bm_result = perform_brunnermunzel_test(group1, group2)
#     # result = perform_wilcoxon_test(group1, group2)
#     # result = cliffs_delta(group1, group2, verbose=True)
#     # result = moods_median_test_two_groups(group1, group2)
    resultbootstrap_ttest = bootstrap_ttest(group1, group2, n_bootstrap=9999, alternative='two-sided')
#
#
#
#     # 3. 打印结果      执行Mann-Whitney U检验
#     print("\n===== Mann-Whitney U检验结果 =====")
#     print(f"U统计量: {result1['U_statistic']:.2f}")
#     print(f"p值: {result1['p_value']:.30f}")
#     print(f"效应量(Cliff's Delta): {result1['cliffs_delta']:.3f}")
#     print(f"组1中位数: {result1['group1_median']:.2f}")
#     print(f"组2中位数: {result1['group2_median']:.2f}")
#     print(f"结论: {result1['interpretation']}")
#
#
#     # # 打印结果       Kolmogorov-Smirnov检验
#     print("===== Kolmogorov-Smirnov检验结果 =====")
#     print(f"KS统计量 D: {ks_result['statistic']:.4f}")
#     print(f"p值: {ks_result['p_value']:.4f}")
#     print(f"效应量 (D值): {ks_result['effect_size']:.4f}")
#     print(f"组1中位数: {ks_result['group1_median']:.4f}")
#     print(f"组2中位数: {ks_result['group2_median']:.4f}")
#     print(f"结论: {ks_result['interpretation']}")
#
#
#
#     # ###brunnermunze
#     print("=====brunnermunze检验结果 =====")
#     print(f"统计量 W: {bm_result['statistic']:.4f}")
#     print(f"p值: {bm_result['p_value']:.4f}")
#     print(f"效应量: {bm_result['effect_size']:.4f}")
#     print(f"组1中位数: {bm_result['group1_median']:.4f}")
#     print(f"组2中位数: {bm_result['group2_median']:.4f}")
#     print(f"结论: {bm_result['interpretation']}")
#
#
    print("===== Bootstrap t检验结果 =====")
    print(f"原始t统计量: {resultbootstrap_ttest['t_original']:.4f}")
    print(f"p值 ({resultbootstrap_ttest['alternative']}): {resultbootstrap_ttest['p_value']:.40f}")
    print(f"均值差异 (组1 - 组2): {resultbootstrap_ttest['mean_diff']:.4f}")
    print(f"95%置信区间: [{resultbootstrap_ttest['ci_lower']:.4f}, {resultbootstrap_ttest['ci_upper']:.4f}]")
    print(f"结论: {resultbootstrap_ttest['interpretation']}")
#
#
#
#
# #     # #########wilcoxon   不要
# #     # print(f"统计量: {result['statistic']}")
# #     # print(f"p值: {result['p_value']:.4f}")
# #     # print(f"效应量 (r): {result['effect_size']:.3f}")
# #     # print(f"中位数差值 (治疗后-前): {result['median_diff']:.3f}")
# #     # print(f"均值差值 (治疗后-前): {result['mean_diff']:.3f}")
# #     # print(f"结论: {result['interpretation']}")
# #     # print("\n描述性统计:")
# #     # print(f"  治疗前中位数: {result['descriptive_stats']['group1_median']:.2f}")
# #     # print(f"  治疗后中位数: {result['descriptive_stats']['group2_median']:.2f}")
# #     # print(f"  配对样本量: {result['descriptive_stats']['n_pairs']}")
# #
# #
# #
# #     ###########  Cliff's Delta 效应量不要
# #     # print("\n===== Cliff's Delta 效应量检验 =====")
# #     # print(f"效应量 δ: {result['delta']:.3f}")
# #     # print(f"效应大小: {result['magnitude']}")
# #     # print(f"统计解释: {result['interpretation']}")
# #     # print(f"优势比较: {result['comparison']}")
# #     # print(f"数据重叠比例: {result['overlap']:.1%}")
# #





