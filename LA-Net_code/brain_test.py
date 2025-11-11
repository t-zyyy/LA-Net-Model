##############原始，不加性别标签
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import torch

class BrainS18(Dataset):
    def __init__(self, root_dir, excel_file, sets):
        """
        初始化类，加载文件名和年龄标签
        :param root_dir: 数据集根目录
        :param excel_file: 包含文件名和年龄标签的 Excel 文件路径
        :param sets: 参数设置
        """
        self.data_info = pd.read_excel(excel_file)  # 读取 Excel 文件
        self.root_dir = root_dir  # 数据集根目录
        self.input_D = sets.input_D  # 输入深度
        self.input_H = sets.input_H  # 输入高度
        self.input_W = sets.input_W  # 输入宽度
        self.phase = sets.phase  # 阶段（train/test）
        print(f"Processing {len(self.data_info)} samples.")  # 打印样本数量

    def __nii2tensorarray__(self, data):
        """
        将 NIfTI 数据转换为 Tensor 格式
        :param data: 输入数据
        :return: 转换后的数据
        """
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])  # 添加通道维度
        new_data = new_data.astype("float32")  # 转为 float32 类型
        return new_data

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.data_info)

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        :param idx: 样本索引
        :return: 处理后的图像和对应的年龄标签
        """
        file_info = self.data_info.iloc[idx]  # 获取当前样本信息
        img_name = os.path.join(self.root_dir, file_info['A'])  # 图像路径
        age_label = file_info['B']  # 年龄标签
        age_label = torch.tensor(age_label).float()  # 将age_label转为torch张量并转换为float类型

        assert os.path.isfile(img_name), f"File {img_name} not found."  # 确保文件存在

        # 加载图像
        #img = nibabel.load(img_name)
        img = np.load(img_name)
        img_array = self.__data_process__(img)  # 处理图像数据

        # 转换为 Tensor 格式
        img_array = self.__nii2tensorarray__(img_array)

        return img_array, age_label

    def __itensity_normalize_one_volume__(self, volume):
        """
        对体积数据进行归一化
        :param volume: 输入体积数据
        :return: 归一化后的数据
        """
        pixels = volume[volume > 0]  # 提取非零像素
        mean = pixels.mean()  # 计算均值
        std = pixels.std()  # 计算标准差
        out = (volume - mean) / std  # 标准化
        out_random = np.random.normal(0, 1, size=volume.shape)  # 为零值生成随机数
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        调整数据大小到网络输入尺寸
        :param data: 输入数据
        :return: 调整大小后的数据
        """
        [depth, height, width] = data.shape
        scale = [self.input_D / depth, self.input_H / height, self.input_W / width]
        data = ndimage.zoom(data, scale, order=0)  # 使用最近邻插值缩放
        return data

    def __data_process__(self, data):
        """
        图像数据预处理
        :param data: 输入 NIfTI 数据
        :return: 处理后的数据
        """
        #data = data.get_fdata()  # 提取图像数据
        data = self.__resize_data__(data)  # 调整尺寸
        data = self.__itensity_normalize_one_volume__(data)  # 归一化
        return data

if __name__ == '__main__':
    print()




# ############处理纯肺
# import os
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset
# import nibabel
# from scipy import ndimage
# import torch
# class BrainS18(Dataset):
#     def __init__(self, root_dir, excel_file, sets):
#         """
#         初始化类，加载文件名和年龄标签
#         :param root_dir: 数据集根目录
#         :param excel_file: 包含文件名和年龄标签的 Excel 文件路径
#         :param sets: 参数设置
#         """
#         self.data_info = pd.read_excel(excel_file)  # 读取 Excel 文件
#         self.root_dir = root_dir  # 数据集根目录
#         self.input_D = sets.input_D  # 输入深度
#         self.input_H = sets.input_H  # 输入高度
#         self.input_W = sets.input_W  # 输入宽度
#         self.phase = sets.phase  # 阶段（train/test）
#         print(f"Processing {len(self.data_info)} samples.")  # 打印样本数量
#
#     def __nii2tensorarray__(self, data):
#         """
#         将 NIfTI 数据转换为 Tensor 格式
#         :param data: 输入数据
#         :return: 转换后的数据
#         """
#         [z, y, x] = data.shape
#         new_data = np.reshape(data, [1, z, y, x])  # 添加通道维度
#         new_data = new_data.astype("float32")  # 转为 float32 类型
#         return new_data
#
#     def __len__(self):
#         """
#         返回数据集大小
#         """
#         return len(self.data_info)
#
#     def __getitem__(self, idx):
#         """
#         获取数据集中的一个样本
#         :param idx: 样本索引
#         :return: 处理后的图像和对应的年龄标签
#         """
#         file_info = self.data_info.iloc[idx]  # 获取当前样本信息
#         img_name = os.path.join(self.root_dir, file_info['A'])  # 图像路径
#         age_label = file_info['B']  # 年龄标签
#         age_label = torch.tensor(age_label).float()  # 将age_label转为torch张量并转换为float类型
#
#         assert os.path.isfile(img_name), f"File {img_name} not found."  # 确保文件存在
#
#         # 加载图像
#         #img = nibabel.load(img_name)
#         img = np.load(img_name)
#         img_array = self.__data_process__(img,img_name)  # 处理图像数据
#
#         # 转换为 Tensor 格式
#         img_array = self.__nii2tensorarray__(img_array)
#
#         return img_array, age_label
#
#     def __itensity_normalize_one_volume__(self, volume,img_name=None):
#         """
#         对体积数据进行归一化
#         :param volume: 输入体积数据
#         :return: 归一化后的数据
#         """
#         pixels = volume[volume > 0]  # 提取非零像素
#
#         if len(pixels) == 0:
#             warning_msg = f"Warning: Empty volume detected in file {img_name}"
#             print(warning_msg)
#             # You could also log this to a file if needed
#             # with open("empty_volumes.log", "a") as f:
#             #     f.write(warning_msg + "\n")
#
#             # Return random noise since we can't normalize
#             return np.random.normal(0, 1, size=volume.shape)
#
#         mean = pixels.mean()  # 计算均值
#         std = pixels.std()  # 计算标准差
#         out = (volume - mean) / std  # 标准化
#         out_random = np.random.normal(0, 1, size=volume.shape)  # 为零值生成随机数
#         out[volume == 0] = out_random[volume == 0]
#         return out
#
#     def __resize_data__(self, data):
#         """
#         调整数据大小到网络输入尺寸
#         :param data: 输入数据
#         :return: 调整大小后的数据
#         """
#         [depth, height, width] = data.shape
#         scale = [self.input_D / depth, self.input_H / height, self.input_W / width]
#         data = ndimage.zoom(data, scale, order=0)  # 使用最近邻插值缩放
#         return data
#
#     def __data_process__(self, data,img_name=None):
#         """
#         图像数据预处理
#         :param data: 输入 NIfTI 数据
#         :return: 处理后的数据
#         """
#         #data = data.get_fdata()  # 提取图像数据
#         data = self.__resize_data__(data)  # 调整尺寸
#         data = self.__itensity_normalize_one_volume__(data,img_name)  # 归一化
#         return data
#
# if __name__ == '__main__':
#     print()










# #######修改，加上性别标签
# import os
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset
# import nibabel
# from scipy import ndimage
# import torch
#
#
# class BrainS18(Dataset):
#     def __init__(self, root_dir, excel_file, sets):
#         """
#         初始化类，加载文件名和年龄标签
#         :param root_dir: 数据集根目录
#         :param excel_file: 包含文件名和年龄标签的 Excel 文件路径
#         :param sets: 参数设置
#         """
#         self.data_info = pd.read_excel(excel_file)  # 读取 Excel 文件
#         self.root_dir = root_dir  # 数据集根目录
#         self.input_D = sets.input_D  # 输入深度
#         self.input_H = sets.input_H  # 输入高度
#         self.input_W = sets.input_W  # 输入宽度
#         self.phase = sets.phase  # 阶段（train/test）
#         print(f"Processing {len(self.data_info)} samples.")  # 打印样本数量
#
#     def __nii2tensorarray__(self, data):
#         """
#         将 NIfTI 数据转换为 Tensor 格式
#         :param data: 输入数据
#         :return: 转换后的数据
#         """
#         [z, y, x] = data.shape
#         new_data = np.reshape(data, [1, z, y, x])  # 添加通道维度
#         new_data = new_data.astype("float32")  # 转为 float32 类型
#         return new_data
#
#     def __len__(self):
#         """
#         返回数据集大小
#         """
#         return len(self.data_info)
#
#     def __getitem__(self, idx):
#         """
#         获取数据集中的一个样本
#         :param idx: 样本索引
#         :return: 处理后的图像和对应的年龄标签
#         """
#         file_info = self.data_info.iloc[idx]  # 获取当前样本信息
#         img_name = os.path.join(self.root_dir, file_info['A'])  # 图像路径
#         assert os.path.isfile(img_name), f"File {img_name} not found."  # 确保文件存在
#
#         # 加载图像
#         img = np.load(img_name)
#         img_array = self.__data_process__(img)  # 处理图像数据
#         img_array = self.__nii2tensorarray__(img_array)  # 转换为 Tensor 格式
#
#         # 加载标签
#         age_label = torch.tensor(file_info['B']).float()  # 年龄
#         sex_label = torch.tensor(file_info['C']).float()  # 性别（0/1）
#         fvc_label = torch.tensor(file_info['D']).float()  # 校正FVC
#         fev1_label = torch.tensor(file_info['E']).float()  # 校正FEV1
#         fev1_ratio_label = torch.tensor(file_info['G']).float()  # 校正FEV1/FEV1_Pred
#
#         return img_array, age_label, sex_label, fvc_label, fev1_label, fev1_ratio_label
#
#
#
#     def __itensity_normalize_one_volume__(self, volume):
#         """
#         对体积数据进行归一化
#         :param volume: 输入体积数据
#         :return: 归一化后的数据
#         """
#         pixels = volume[volume > 0]  # 提取非零像素
#         mean = pixels.mean()  # 计算均值
#         std = pixels.std()  # 计算标准差
#         out = (volume - mean) / std  # 标准化
#         out_random = np.random.normal(0, 1, size=volume.shape)  # 为零值生成随机数
#         out[volume == 0] = out_random[volume == 0]
#         return out
#
#     def __resize_data__(self, data):
#         """
#         调整数据大小到网络输入尺寸
#         :param data: 输入数据
#         :return: 调整大小后的数据
#         """
#         [depth, height, width] = data.shape
#         scale = [self.input_D / depth, self.input_H / height, self.input_W / width]
#         data = ndimage.zoom(data, scale, order=0)  # 使用最近邻插值缩放
#         return data
#
#     def __data_process__(self, data):
#         """
#         图像数据预处理
#         :param data: 输入 NIfTI 数据
#         :return: 处理后的数据
#         """
#         # data = data.get_fdata()  # 提取图像数据
#         data = self.__resize_data__(data)  # 调整尺寸
#         data = self.__itensity_normalize_one_volume__(data)  # 归一化
#         return data
#
#
# if __name__ == '__main__':
#     print()