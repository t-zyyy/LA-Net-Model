##############原始，不加性别标签
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy import ndimage
import torch
# from setting import parse_opt

from setting import parse_opts  # 从 config.py 导入函数

args = parse_opts()

class BrainS18Dataset(Dataset):
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
        self.augment = sets.phase == "train"  # 仅在训练阶段使用数据增强
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
        # print("file_info['A'] =", file_info['A'], "类型 =", type(file_info['A']))

        img_name = os.path.join(self.root_dir, file_info['A'])  # 图像路径
        age_label = file_info['B']  # 年龄标签
        age_label = torch.tensor(age_label).float()  # 转为 float 类型的 Tensor

        assert os.path.isfile(img_name), f"File {img_name} not found."  # 确保文件存在

        # 加载图像
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




#####2222222222
    def __data_augmentation__(self, data):
        """
        针对CT图像的优化数据增强方法：50%概率引入增强，每次随机组合一种或多种增强。
        输入大小为 (60, 224, 224)。
        """
        if np.random.rand() >= 0:
            num_augmentations = np.random.randint(1, 3)  # 随机选择1到2种增强方法
            augmentations = np.random.choice(["flip", "rotate", "noise", "crop_pad"], num_augmentations, replace=False)

            for augmentation_choice in augmentations:
                if augmentation_choice == "flip":
                    # 随机在高度或宽度方向翻转，保持深度不变
                    # axis = np.random.choice([1, 2])  # 不翻转深度轴（即不选择0）
                    # data = np.flip(data, axis=axis)
                    pass   #我觉的翻转会破坏结构

                elif augmentation_choice == "rotate":
                    # 随机选择旋转轴和角度，限制旋转角度以保护解剖结构
                    axes = [(0, 1), (0, 2), (1, 2)]  # 定义所有可能的旋转轴组合
                    chosen_axes = np.random.choice(len(axes), 1)[0]# 从轴的索引中随机选择
                    chosen_axes = axes[chosen_axes]  # 使用选择的索引从 axes 中取出具体的轴

                    angle = np.random.uniform(-15, 15)  # 缩小角度范围至±15度
                    data = ndimage.rotate(data, angle, axes=chosen_axes, reshape=False, mode='reflect')

                elif augmentation_choice == "noise":
                    # 随机选择1-2种噪声类型
                    num_noise_types = np.random.randint(1, 3)  # 随机选择1到2种噪声类型
                    noise_types = np.random.choice(["salt_pepper", "multiplicative", "gaussian", "poisson"],
                                                   num_noise_types, replace=False)

                    for noise_choice in noise_types:
                        if noise_choice == "salt_pepper":
                            # 椒盐噪声
                            salt_pepper_ratio = 0.003  # 椒盐噪声的比例
                            noise_mask = np.random.choice([0, 1, 2], size=data.shape,
                                                          p=[1 - salt_pepper_ratio, salt_pepper_ratio / 2,
                                                             salt_pepper_ratio / 2])
                            data[noise_mask == 1] = data.max()  # 盐噪声
                            data[noise_mask == 2] = data.min()  # 椒噪声

                        elif noise_choice == "multiplicative":
                            # 乘性噪声
                            multiplicative_noise = np.random.uniform(0.97, 1.03, size=data.shape)  # 随机缩放因子
                            data *= multiplicative_noise

                        elif noise_choice == "gaussian":
                            # 高斯噪声
                            gaussian_noise = np.random.normal(0, 0.01, size=data.shape)  # 高斯噪声的标准差
                            data += gaussian_noise

                        elif noise_choice == "poisson":
                            # 泊松噪声
                            scale_factor = 1e5  # 可根据实际情况调整噪声强度

                            # 确保数据没有负值
                            data = np.clip(data, a_min=0, a_max=None)

                            # 确保数据不包含 NaN
                            if np.any(np.isnan(data)):
                                data = np.nan_to_num(data)  # 将 NaN 值替换为 0 或其他合理值

                            # 添加泊松噪声
                            poisson_noise = np.random.poisson(data * scale_factor) / scale_factor - data
                            data += poisson_noise


                    # 确保数据值在合理范围内
                    data = np.clip(data, a_min=data.min(), a_max=data.max())

                elif augmentation_choice == "crop_pad":
                    # 随机裁剪或填充深度方向，确保最终尺寸为 (60, 224, 224)
                    max_crop_ratio = 0.1
                    depth, height, width = data.shape
                    max_crop = int(max_crop_ratio * depth)
                    crop = np.random.randint(-max_crop, max_crop + 1)  # 允许增加或减少

                    if crop > 0:  # 从头裁剪并填充
                        data = data[crop:]
                        data = np.pad(data, ((crop, 0), (0, 0), (0, 0)), mode="reflect")
                    elif crop < 0:  # 从尾裁剪并填充
                        data = data[:crop]
                        data = np.pad(data, ((0, -crop), (0, 0), (0, 0)), mode="reflect")

                    # 确保输出尺寸正确
                    # if data.shape[0] != 60:


                    #     diff = 60 - data.shape[0]
                    #     if diff > 0:
                    #         data = np.pad(data, ((diff, 0), (0, 0), (0, 0)), mode="reflect")
                    #     else:
                    #         data = data[-60:]
                    if data.shape[0] != 60:
                        diff = 60 - data.shape[0]
                        if diff > 0:
                            data = np.pad(data, ((diff, 0), (0, 0), (0, 0)), mode="reflect")
                        else:
                            data = data[-60:]

            # 数据归一化
            data = (data - data.min()) / (data.max() - data.min())

        return data








##############处理调用__data_augmentation__的核心

    def __data_process__(self, data):
        """
        图像数据预处理
        :param data: 输入 NIfTI 数据
        :return: 处理后的数据
        """
        data = self.__resize_data__(data)  # 调整尺寸

        if self.augment and np.random.rand() >= 0.9:  # 随机决定是否进行数据增强
            data = self.__data_augmentation__(data)

        data = self.__itensity_normalize_one_volume__(data)  # 归一化

        return data

if __name__ == '__main__':
    print()










# ############处理纯肺
# import os
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset
# from scipy import ndimage
# import torch
# # from setting import parse_opt
#
# from setting import parse_opts  # 从 config.py 导入函数
#
# args = parse_opts()
#
# class BrainS18Dataset(Dataset):
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
#         self.augment = sets.phase == "train"  # 仅在训练阶段使用数据增强
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
#
#
#         file_info = self.data_info.iloc[idx]  # 获取当前样本信息
#         # print("file_info['A'] =", file_info['A'], "类型 =", type(file_info['A']))
#
#         img_name = os.path.join(self.root_dir, file_info['A'])  # 图像路径
#         age_label = file_info['B']  # 年龄标签
#         age_label = torch.tensor(age_label).float()  # 转为 float 类型的 Tensor
#
#         assert os.path.isfile(img_name), f"File {img_name} not found."  # 确保文件存在
#
#         # 加载图像
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
#
#
#
# #####2222222222
#     def __data_augmentation__(self, data):
#         """
#         针对CT图像的优化数据增强方法：50%概率引入增强，每次随机组合一种或多种增强。
#         输入大小为 (60, 224, 224)。
#         """
#         if np.random.rand() >= 0:
#             num_augmentations = np.random.randint(1, 3)  # 随机选择1到2种增强方法
#             augmentations = np.random.choice(["flip", "rotate", "noise", "crop_pad"], num_augmentations, replace=False)
#
#             for augmentation_choice in augmentations:
#                 if augmentation_choice == "flip":
#                     # 随机在高度或宽度方向翻转，保持深度不变
#                     # axis = np.random.choice([1, 2])  # 不翻转深度轴（即不选择0）
#                     # data = np.flip(data, axis=axis)
#                     pass   #我觉的翻转会破坏结构
#
#                 elif augmentation_choice == "rotate":
#                     # 随机选择旋转轴和角度，限制旋转角度以保护解剖结构
#                     axes = [(0, 1), (0, 2), (1, 2)]  # 定义所有可能的旋转轴组合
#                     chosen_axes = np.random.choice(len(axes), 1)[0]# 从轴的索引中随机选择
#                     chosen_axes = axes[chosen_axes]  # 使用选择的索引从 axes 中取出具体的轴
#
#                     angle = np.random.uniform(-15, 15)  # 缩小角度范围至±15度
#                     data = ndimage.rotate(data, angle, axes=chosen_axes, reshape=False, mode='reflect')
#
#                 elif augmentation_choice == "noise":
#                     # 随机选择1-2种噪声类型
#                     num_noise_types = np.random.randint(1, 3)  # 随机选择1到2种噪声类型
#                     noise_types = np.random.choice(["salt_pepper", "multiplicative", "gaussian", "poisson"],
#                                                    num_noise_types, replace=False)
#
#                     for noise_choice in noise_types:
#                         if noise_choice == "salt_pepper":
#                             # 椒盐噪声
#                             salt_pepper_ratio = 0.003  # 椒盐噪声的比例
#                             noise_mask = np.random.choice([0, 1, 2], size=data.shape,
#                                                           p=[1 - salt_pepper_ratio, salt_pepper_ratio / 2,
#                                                              salt_pepper_ratio / 2])
#                             data[noise_mask == 1] = data.max()  # 盐噪声
#                             data[noise_mask == 2] = data.min()  # 椒噪声
#
#                         elif noise_choice == "multiplicative":
#                             # 乘性噪声
#                             multiplicative_noise = np.random.uniform(0.97, 1.03, size=data.shape)  # 随机缩放因子
#                             data *= multiplicative_noise
#
#                         elif noise_choice == "gaussian":
#                             # 高斯噪声
#                             gaussian_noise = np.random.normal(0, 0.01, size=data.shape)  # 高斯噪声的标准差
#                             data += gaussian_noise
#
#                         elif noise_choice == "poisson":
#                             # 泊松噪声
#                             scale_factor = 1e5  # 可根据实际情况调整噪声强度
#
#                             # 确保数据没有负值
#                             data = np.clip(data, a_min=0, a_max=None)
#
#                             # 确保数据不包含 NaN
#                             if np.any(np.isnan(data)):
#                                 data = np.nan_to_num(data)  # 将 NaN 值替换为 0 或其他合理值
#
#                             # 添加泊松噪声
#                             poisson_noise = np.random.poisson(data * scale_factor) / scale_factor - data
#                             data += poisson_noise
#
#
#                     # 确保数据值在合理范围内
#                     data = np.clip(data, a_min=data.min(), a_max=data.max())
#
#                 elif augmentation_choice == "crop_pad":
#                     # 随机裁剪或填充深度方向，确保最终尺寸为 (60, 224, 224)
#                     max_crop_ratio = 0.1
#                     depth, height, width = data.shape
#                     max_crop = int(max_crop_ratio * depth)
#                     crop = np.random.randint(-max_crop, max_crop + 1)  # 允许增加或减少
#
#                     if crop > 0:  # 从头裁剪并填充
#                         data = data[crop:]
#                         data = np.pad(data, ((crop, 0), (0, 0), (0, 0)), mode="reflect")
#                     elif crop < 0:  # 从尾裁剪并填充
#                         data = data[:crop]
#                         data = np.pad(data, ((0, -crop), (0, 0), (0, 0)), mode="reflect")
#
#                     # 确保输出尺寸正确
#                     # if data.shape[0] != 60:
#
#
#                     #     diff = 60 - data.shape[0]
#                     #     if diff > 0:
#                     #         data = np.pad(data, ((diff, 0), (0, 0), (0, 0)), mode="reflect")
#                     #     else:
#                     #         data = data[-60:]
#                     if data.shape[0] != 60:
#                         diff = 60 - data.shape[0]
#                         if diff > 0:
#                             data = np.pad(data, ((diff, 0), (0, 0), (0, 0)), mode="reflect")
#                         else:
#                             data = data[-60:]
#
#             # 数据归一化
#             data = (data - data.min()) / (data.max() - data.min())
#
#         return data
#
#
#
#
#
#
#
#
# ##############处理调用__data_augmentation__的核心
#
#     def __data_process__(self, data,img_name=None):
#         """
#         图像数据预处理
#         :param data: 输入 NIfTI 数据
#         :return: 处理后的数据
#         """
#         data = self.__resize_data__(data)  # 调整尺寸
#
#         if self.augment and np.random.rand() >= 0.9:  # 随机决定是否进行数据增强
#             data = self.__data_augmentation__(data)
#
#         data = self.__itensity_normalize_one_volume__(data,img_name)  # 归一化
#
#         return data
#
# if __name__ == '__main__':
#     print()








#
# #######修改，加上性别标签
# import os
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset
# import nibabel
# from scipy import ndimage
# import torch
#
# class BrainS18Dataset(Dataset):
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
#         self.augment = sets.phase == "train"  # 仅在训练阶段使用数据增强
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
#         age_label = torch.tensor(age_label).float()  # 转为 float 类型的 Tensor
#         sex_label = file_info['C']
#         sex_label = torch.tensor(sex_label).float()  # 如果性别是数字型（如0/1），否则需要先转换
#
#         assert os.path.isfile(img_name), f"File {img_name} not found."  # 确保文件存在
#
#         # 加载图像
#         img = np.load(img_name)
#         img_array = self.__data_process__(img)  # 处理图像数据
#
#         # 转换为 Tensor 格式
#         img_array = self.__nii2tensorarray__(img_array)
#
#         return img_array, age_label, sex_label
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
#
#
#
# #####2222222222
#     def __data_augmentation__(self, data):
#         """
#         针对CT图像的优化数据增强方法：50%概率引入增强，每次随机组合一种或多种增强。
#         输入大小为 (60, 224, 224)。
#         """
#         if np.random.rand() >= 0:
#             num_augmentations = np.random.randint(1, 3)  # 随机选择1到2种增强方法
#             augmentations = np.random.choice(["flip", "rotate", "noise", "crop_pad"], num_augmentations, replace=False)
#
#             for augmentation_choice in augmentations:
#                 if augmentation_choice == "flip":
#                     # 随机在高度或宽度方向翻转，保持深度不变
#                     # axis = np.random.choice([1, 2])  # 不翻转深度轴（即不选择0）
#                     # data = np.flip(data, axis=axis)
#                     pass   #我觉的翻转会破坏结构
#
#                 elif augmentation_choice == "rotate":
#                     # 随机选择旋转轴和角度，限制旋转角度以保护解剖结构
#                     axes = [(0, 1), (0, 2), (1, 2)]  # 定义所有可能的旋转轴组合
#                     chosen_axes = np.random.choice(len(axes), 1)[0]# 从轴的索引中随机选择
#                     chosen_axes = axes[chosen_axes]  # 使用选择的索引从 axes 中取出具体的轴
#
#                     angle = np.random.uniform(-15, 15)  # 缩小角度范围至±15度
#                     data = ndimage.rotate(data, angle, axes=chosen_axes, reshape=False, mode='reflect')
#
#                 elif augmentation_choice == "noise":
#                     # 随机选择1-2种噪声类型
#                     num_noise_types = np.random.randint(1, 3)  # 随机选择1到2种噪声类型
#                     noise_types = np.random.choice(["salt_pepper", "multiplicative", "gaussian", "poisson"],
#                                                    num_noise_types, replace=False)
#
#                     for noise_choice in noise_types:
#                         if noise_choice == "salt_pepper":
#                             # 椒盐噪声
#                             salt_pepper_ratio = 0.003  # 椒盐噪声的比例
#                             noise_mask = np.random.choice([0, 1, 2], size=data.shape,
#                                                           p=[1 - salt_pepper_ratio, salt_pepper_ratio / 2,
#                                                              salt_pepper_ratio / 2])
#                             data[noise_mask == 1] = data.max()  # 盐噪声
#                             data[noise_mask == 2] = data.min()  # 椒噪声
#
#                         elif noise_choice == "multiplicative":
#                             # 乘性噪声
#                             multiplicative_noise = np.random.uniform(0.97, 1.03, size=data.shape)  # 随机缩放因子
#                             data *= multiplicative_noise
#
#                         elif noise_choice == "gaussian":
#                             # 高斯噪声
#                             gaussian_noise = np.random.normal(0, 0.01, size=data.shape)  # 高斯噪声的标准差
#                             data += gaussian_noise
#
#                         elif noise_choice == "poisson":
#                             # 泊松噪声
#                             scale_factor = 1e5  # 可根据实际情况调整噪声强度
#
#                             # 确保数据没有负值
#                             data = np.clip(data, a_min=0, a_max=None)
#
#                             # 确保数据不包含 NaN
#                             if np.any(np.isnan(data)):
#                                 data = np.nan_to_num(data)  # 将 NaN 值替换为 0 或其他合理值
#
#                             # 添加泊松噪声
#                             poisson_noise = np.random.poisson(data * scale_factor) / scale_factor - data
#                             data += poisson_noise
#
#
#                     # 确保数据值在合理范围内
#                     data = np.clip(data, a_min=data.min(), a_max=data.max())
#
#                 elif augmentation_choice == "crop_pad":
#                     # 随机裁剪或填充深度方向，确保最终尺寸为 (60, 224, 224)
#                     max_crop_ratio = 0.1
#                     depth, height, width = data.shape
#                     max_crop = int(max_crop_ratio * depth)
#                     crop = np.random.randint(-max_crop, max_crop + 1)  # 允许增加或减少
#
#                     if crop > 0:  # 从头裁剪并填充
#                         data = data[crop:]
#                         data = np.pad(data, ((crop, 0), (0, 0), (0, 0)), mode="reflect")
#                     elif crop < 0:  # 从尾裁剪并填充
#                         data = data[:crop]
#                         data = np.pad(data, ((0, -crop), (0, 0), (0, 0)), mode="reflect")
#
#                     # 确保输出尺寸正确
#                     if data.shape[0] != 60:
#                         diff = 60 - data.shape[0]
#                         if diff > 0:
#                             data = np.pad(data, ((diff, 0), (0, 0), (0, 0)), mode="reflect")
#                         else:
#                             data = data[-60:]
#
#             # 数据归一化
#             data = (data - data.min()) / (data.max() - data.min())
#
#         return data
#
#
#
#
#
#
#
#
# ##############处理调用__data_augmentation__的核心
#
#     def __data_process__(self, data):
#         """
#         图像数据预处理
#         :param data: 输入 NIfTI 数据
#         :return: 处理后的数据
#         """
#         data = self.__resize_data__(data)  # 调整尺寸
#
#         if self.augment and np.random.rand() >= 0.85:  # 随机决定是否进行数据增强
#             data = self.__data_augmentation__(data)
#
#         data = self.__itensity_normalize_one_volume__(data)  # 归一化
#
#         return data
#
# if __name__ == '__main__':
#     print()
#





#
# ##########多任务
#
# import os
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset
# import nibabel
# from scipy import ndimage
# import torch
#
# class BrainS18Dataset(Dataset):
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
#         self.augment = sets.phase == "train"  # 仅在训练阶段使用数据增强
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
#
#
#
# #####2222222222
#     def __data_augmentation__(self, data):
#         """
#         针对CT图像的优化数据增强方法：50%概率引入增强，每次随机组合一种或多种增强。
#         输入大小为 (60, 224, 224)。
#         """
#         if np.random.rand() >= 0:
#             num_augmentations = np.random.randint(1, 3)  # 随机选择1到2种增强方法
#             augmentations = np.random.choice(["flip", "rotate", "noise", "crop_pad"], num_augmentations, replace=False)
#
#             for augmentation_choice in augmentations:
#                 if augmentation_choice == "flip":
#                     # 随机在高度或宽度方向翻转，保持深度不变
#                     # axis = np.random.choice([1, 2])  # 不翻转深度轴（即不选择0）
#                     # data = np.flip(data, axis=axis)
#                     pass   #我觉的翻转会破坏结构
#
#                 elif augmentation_choice == "rotate":
#                     # 随机选择旋转轴和角度，限制旋转角度以保护解剖结构
#                     axes = [(0, 1), (0, 2), (1, 2)]  # 定义所有可能的旋转轴组合
#                     chosen_axes = np.random.choice(len(axes), 1)[0]# 从轴的索引中随机选择
#                     chosen_axes = axes[chosen_axes]  # 使用选择的索引从 axes 中取出具体的轴
#
#                     angle = np.random.uniform(-15, 15)  # 缩小角度范围至±15度
#                     data = ndimage.rotate(data, angle, axes=chosen_axes, reshape=False, mode='reflect')
#
#                 elif augmentation_choice == "noise":
#                     # 随机选择1-2种噪声类型
#                     num_noise_types = np.random.randint(1, 3)  # 随机选择1到2种噪声类型
#                     noise_types = np.random.choice(["salt_pepper", "multiplicative", "gaussian", "poisson"],
#                                                    num_noise_types, replace=False)
#
#                     for noise_choice in noise_types:
#                         if noise_choice == "salt_pepper":
#                             # 椒盐噪声
#                             salt_pepper_ratio = 0.003  # 椒盐噪声的比例
#                             noise_mask = np.random.choice([0, 1, 2], size=data.shape,
#                                                           p=[1 - salt_pepper_ratio, salt_pepper_ratio / 2,
#                                                              salt_pepper_ratio / 2])
#                             data[noise_mask == 1] = data.max()  # 盐噪声
#                             data[noise_mask == 2] = data.min()  # 椒噪声
#
#                         elif noise_choice == "multiplicative":
#                             # 乘性噪声
#                             multiplicative_noise = np.random.uniform(0.97, 1.03, size=data.shape)  # 随机缩放因子
#                             data *= multiplicative_noise
#
#                         elif noise_choice == "gaussian":
#                             # 高斯噪声
#                             gaussian_noise = np.random.normal(0, 0.01, size=data.shape)  # 高斯噪声的标准差
#                             data += gaussian_noise
#
#                         elif noise_choice == "poisson":
#                             # 泊松噪声
#                             scale_factor = 1e5  # 可根据实际情况调整噪声强度
#
#                             # 确保数据没有负值
#                             data = np.clip(data, a_min=0, a_max=None)
#
#                             # 确保数据不包含 NaN
#                             if np.any(np.isnan(data)):
#                                 data = np.nan_to_num(data)  # 将 NaN 值替换为 0 或其他合理值
#
#                             # 添加泊松噪声
#                             poisson_noise = np.random.poisson(data * scale_factor) / scale_factor - data
#                             data += poisson_noise
#
#
#                     # 确保数据值在合理范围内
#                     data = np.clip(data, a_min=data.min(), a_max=data.max())
#
#                 elif augmentation_choice == "crop_pad":
#                     # 随机裁剪或填充深度方向，确保最终尺寸为 (60, 224, 224)
#                     max_crop_ratio = 0.1
#                     depth, height, width = data.shape
#                     max_crop = int(max_crop_ratio * depth)
#                     crop = np.random.randint(-max_crop, max_crop + 1)  # 允许增加或减少
#
#                     if crop > 0:  # 从头裁剪并填充
#                         data = data[crop:]
#                         data = np.pad(data, ((crop, 0), (0, 0), (0, 0)), mode="reflect")
#                     elif crop < 0:  # 从尾裁剪并填充
#                         data = data[:crop]
#                         data = np.pad(data, ((0, -crop), (0, 0), (0, 0)), mode="reflect")
#
#                     # 确保输出尺寸正确
#                     if data.shape[0] != 60:
#                         diff = 60 - data.shape[0]
#                         if diff > 0:
#                             data = np.pad(data, ((diff, 0), (0, 0), (0, 0)), mode="reflect")
#                         else:
#                             data = data[-60:]
#
#             # 数据归一化
#             data = (data - data.min()) / (data.max() - data.min())
#
#         return data
#
#
#
#
#
#
#
#
# ##############处理调用__data_augmentation__的核心
#
#     def __data_process__(self, data):
#         """
#         图像数据预处理
#         :param data: 输入 NIfTI 数据
#         :return: 处理后的数据
#         """
#         data = self.__resize_data__(data)  # 调整尺寸
#
#         if self.augment and np.random.rand() >= 1:  # 随机决定是否进行数据增强
#             data = self.__data_augmentation__(data)
#
#         data = self.__itensity_normalize_one_volume__(data)  # 归一化
#
#         return data
#
# if __name__ == '__main__':
#     print()
#
