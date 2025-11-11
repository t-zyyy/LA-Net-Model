
import numpy as np
from torch.utils.data import DataLoader
from setting import parse_opts
from brains import BrainS18Dataset
from model import generate_model
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入 tqdm 库来显示进度条
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 使系统智能看到编号为3的GPU
import torch

# 设置为避免不必要的警告
np.seterr(all='ignore')

def run_ce(data_loader, model, img_names, sets, labels):
    """
    测试回归模型，并计算预测值与真实值的差异。
    """
    predictions = []
    model.eval()  # 切换到评估模式

    # 使用 tqdm 显示进度条
    with tqdm(data_loader, desc="Processing Samples", unit="batch") as pbar:
        for batch_id, batch_data in enumerate(pbar):
            # 获取输入数据
            volume, _ = batch_data  # 假设 batch_data 包含图像和标签

            # 如果 volume 是一个列表或其他非 Tensor 类型的数据
            if isinstance(volume, list):
                volume = torch.tensor(volume, dtype=torch.float32)  # 转换为 Tensor
            elif isinstance(volume, np.ndarray):
                volume = torch.tensor(volume, dtype=torch.float32)  # 如果是 numpy 数组，转换为 Tensor

            # 确保数据是一个 Tensor 类型
            if not isinstance(volume, torch.Tensor):
                raise TypeError(f"Expected volume to be a Tensor, but got {type(volume)}")

            if not sets.no_cuda:
                volume = volume.cuda()  # 将数据移动到 GPU

            with torch.no_grad():
                pred = model(volume)  # 获取预测值
                pred = pred.cpu().numpy().flatten()  # 转为CPU并展平为数组
                predictions.append(pred[0])  # 保留每次批次的预测结果

            # 使用 tqdm.write() 输出每个批次的预测值而不覆盖进度条
            tqdm.write(f"Batch {batch_id + 1}: Predicted Age = {pred[0]:.2f}, True Age = {labels[batch_id]:.2f}")

    # 打印预测结果和真实标签
    for idx, pred in enumerate(predictions):
        print(f"Image: {img_names[idx]} | Predicted Age: {pred:.2f} | True Age: {labels[idx]}")

    return predictions




def visualize_results(predictions, labels, mean_error, train_error):
    """
    可视化预测值与真实值的对比，并绘制误差分布。
    包含Mean Absolute Error (MAE)作为图的标题。
    """
    # 1. 绘制预测值与真实值的散点图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(labels, predictions, color='blue', alpha=0.6)
    plt.plot([min(labels), max(labels)], [min(labels), max(labels)], color='red', linestyle='--')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title(f'True vs Predicted Age\ntrain Absolute Error (MAE): {train_error:.2f}')

    # 2. 绘制误差的直方图
    errors = [abs(pred - true) for pred, true in zip(predictions, labels)]
    n = len(errors)  # 获取误差的数量
    weights = np.ones_like(errors) / n * 100  # 每个数据点的权重为 1/n，并乘以 100 转换为百分比

    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=20, weights=weights, color='green', alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency (%)')
    plt.title(f'Error Distribution\nval Absolute Error (MAE): {mean_error:.2f}')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    yu = [0]*40
    # 加载设置
    sets = parse_opts()
    sets.target_type = "age_regression"  # 设置任务类型
    sets.phase = 'test'
    sets.test_path='/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_4(40_45)/resnet_34/epoch_28_batch_1_3.029997.pth.tar'
    sets.input_D = 60
    sets.test_root = '/home/zsq/train/pre_process/test60'
    sets.test_file = '/home/zsq/train/pre_process/DATE/linshi2/age_range_40_45(3.12).xlsx'

    checkpoint = torch.load(sets.test_path, map_location=torch.device('cpu'))  # 加载模型权重

    # 如果模型是使用DataParallel保存的，去除'模module.'前缀
    model_state_dict = checkpoint['state_dict']  # 假设模型权重在 'state_dict' 中
    new_state_dict = {}

    for key, value in model_state_dict.items():
        new_key = key.replace('module.', '')  # 去掉'模module.'前缀
        new_state_dict[new_key] = value

    # 生成模型
    net, _ = generate_model(sets)  # 获取模型和其他返回值（如果有）

    # # 检查模型是否已经是 DataParallel 包装过的
    # if not isinstance(net, torch.nn.DataParallel):
    #     net = torch.nn.DataParallel(net)

    # 加载模型权重
    net.load_state_dict(checkpoint['state_dict'])


    # 加载测试数据集
    n = 232  # 自定义数量，取前 n 个样本
    testing_data = BrainS18Dataset(sets.test_root, sets.test_file, sets)
    subset_data = torch.utils.data.Subset(testing_data, range(n))  # 取前 n 个样本
    data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)

    # 读取图像文件名和对应标签（从 Excel 文件）
    df = pd.read_excel(sets.test_file)  # 读取 Excel 文件
    img_names = df.iloc[:, 0].tolist()  # A列为文件名
    labels = df.iloc[:, 1].tolist()  # B列为标签
    labels = labels[:n]  # 截取前 n 个标签

    # 测试模型并获取预测值
    predictions1 = run_ce(data_loader, net, img_names, sets, labels)
    errors = [abs(pred - true) for pred, true in zip(predictions1, labels)]
    mean_error = np.mean(errors)
    print(mean_error)


    #######################第二个
    sets = parse_opts()
    sets.target_type = "age_regression"  # 设置任务类型
    sets.phase = 'test'
    sets.test_path='/home/zsq/train/MedicalNet_pytorch_files/trails/models/zhengshi/resnet_34/epoch_59_batch_645_3.267063.pth.tar'
    sets.input_D = 60
    sets.test_root='/home/zsq/train/COPD_60'
    sets.test_file = '/home/zsq/train/pre_process/DATE/COPD/COPD(0.6_0.64).xlsx'

    checkpoint = torch.load(sets.test_path, map_location=torch.device('cpu'))  # 加载模型权重

    # 如果模型是使用DataParallel保存的，去除'模module.'前缀
    model_state_dict = checkpoint['state_dict']  # 假设模型权重在 'state_dict' 中
    new_state_dict = {}

    for key, value in model_state_dict.items():
        new_key = key.replace('module.', '')  # 去掉'模module.'前缀
        new_state_dict[new_key] = value

    # 生成模型
    net, _ = generate_model(sets)  # 获取模型和其他返回值（如果有）

    # # 检查模型是否已经是 DataParallel 包装过的
    # if not isinstance(net, torch.nn.DataParallel):
    #     net = torch.nn.DataParallel(net)

    # 加载模型权重
    net.load_state_dict(checkpoint['state_dict'])


    # 加载测试数据集
    n = 179  # 自定义数量，取前 n 个样本
    testing_data = BrainS18Dataset(sets.test_root, sets.test_file, sets)
    subset_data = torch.utils.data.Subset(testing_data, range(n))  # 取前 n 个样本
    data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)

    # 读取图像文件名和对应标签（从 Excel 文件）
    df = pd.read_excel(sets.test_file)  # 读取 Excel 文件
    img_names = df.iloc[:, 0].tolist()  # A列为文件名
    labels = df.iloc[:, 1].tolist()  # B列为标签
    labels = labels[:n]  # 截取前 n 个标签

    # 测试模型并获取预测值
    predictions2 = run_ce(data_loader, net, img_names, sets, labels)
    errors = [abs(pred - true) for pred, true in zip(predictions2, labels)]
    mean_error = np.mean(errors)
    print(mean_error)
    ####################第二个
    #######################第三个
    sets = parse_opts()
    sets.target_type = "age_regression"  # 设置任务类型
    sets.phase = 'test'
    sets.test_path='/home/zsq/train/MedicalNet_pytorch_files/trails/models/zhengshi/resnet_34/epoch_59_batch_645_3.267063.pth.tar'
    sets.input_D = 60
    sets.test_root='/home/zsq/train/COPD_60'
    sets.test_file = '/home/zsq/train/pre_process/DATE/COPD/COPD(0.65_0.69).xlsx'

    checkpoint = torch.load(sets.test_path, map_location=torch.device('cpu'))  # 加载模型权重

    # 如果模型是使用DataParallel保存的，去除'模module.'前缀
    model_state_dict = checkpoint['state_dict']  # 假设模型权重在 'state_dict' 中
    new_state_dict = {}

    for key, value in model_state_dict.items():
        new_key = key.replace('module.', '')  # 去掉'模module.'前缀
        new_state_dict[new_key] = value

    # 生成模型
    net, _ = generate_model(sets)  # 获取模型和其他返回值（如果有）

    # # 检查模型是否已经是 DataParallel 包装过的
    # if not isinstance(net, torch.nn.DataParallel):
    #     net = torch.nn.DataParallel(net)

    # 加载模型权重
    net.load_state_dict(checkpoint['state_dict'])


    # 加载测试数据集
    n = 531  # 自定义数量，取前 n 个样本
    testing_data = BrainS18Dataset(sets.test_root, sets.test_file, sets)
    subset_data = torch.utils.data.Subset(testing_data, range(n))  # 取前 n 个样本
    data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)

    # 读取图像文件名和对应标签（从 Excel 文件）
    df = pd.read_excel(sets.test_file)  # 读取 Excel 文件
    img_names = df.iloc[:, 0].tolist()  # A列为文件名
    labels = df.iloc[:, 1].tolist()  # B列为标签
    labels = labels[:n]  # 截取前 n 个标签

    # 测试模型并获取预测值
    predictions3 = run_ce(data_loader, net, img_names, sets, labels)
    errors = [abs(pred - true) for pred, true in zip(predictions3, labels)]
    mean_error = np.mean(errors)
    print(mean_error)
    ####################第三个
    #######################第四个
    sets = parse_opts()
    sets.target_type = "age_regression"  # 设置任务类型
    sets.phase = 'test'
    sets.test_path='/home/zsq/train/MedicalNet_pytorch_files/trails/models/zhengshi/resnet_34/epoch_59_batch_645_3.267063.pth.tar'
    sets.input_D = 60
    sets.test_root='/home/zsq/train/COPD_60'
    sets.test_file = '/home/zsq/train/pre_process/DATE/COPD/COPD.xlsx'

    checkpoint = torch.load(sets.test_path, map_location=torch.device('cpu'))  # 加载模型权重

    # 如果模型是使用DataParallel保存的，去除'模module.'前缀
    model_state_dict = checkpoint['state_dict']  # 假设模型权重在 'state_dict' 中
    new_state_dict = {}

    for key, value in model_state_dict.items():
        new_key = key.replace('module.', '')  # 去掉'模module.'前缀
        new_state_dict[new_key] = value

    # 生成模型
    net, _ = generate_model(sets)  # 获取模型和其他返回值（如果有）

    # # 检查模型是否已经是 DataParallel 包装过的
    # if not isinstance(net, torch.nn.DataParallel):
    #     net = torch.nn.DataParallel(net)

    # 加载模型权重
    net.load_state_dict(checkpoint['state_dict'])


    # 加载测试数据集
    n = 901  # 自定义数量，取前 n 个样本
    testing_data = BrainS18Dataset(sets.test_root, sets.test_file, sets)
    subset_data = torch.utils.data.Subset(testing_data, range(n))  # 取前 n 个样本
    data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)

    # 读取图像文件名和对应标签（从 Excel 文件）
    df = pd.read_excel(sets.test_file)  # 读取 Excel 文件
    img_names = df.iloc[:, 0].tolist()  # A列为文件名
    labels = df.iloc[:, 1].tolist()  # B列为标签
    labels = labels[:n]  # 截取前 n 个标签

    # 测试模型并获取预测值
    predictions4 = run_ce(data_loader, net, img_names, sets, labels)
    errors = [abs(pred - true) for pred, true in zip(predictions4, labels)]
    mean_error = np.mean(errors)
    print(mean_error)
    ####################第四个


    predictions1 = np.array(predictions1)
    predictions2 = np.array(predictions2)
    predictions3 = np.array(predictions3)
    predictions4 = np.array(predictions4)

    predictions = predictions1*1/4 + predictions2*1/4 + predictions3*1/4 + predictions4*1/4
    # predictions = predictions1
    # 计算平均误差
    errors = [abs(pred - true) for pred, true in zip(predictions, labels)]
    mean_error = np.mean(errors)
    print(mean_error)


    # # 看一下是哪些年龄不准确
    # for label, error in zip(labels, errors):
    #     if(error > 6):
    #         yu[label - 35]+=1
    # # 输出更新后的 yu
    # print("Updated yu:", yu)
    # total = sum(yu)
    # print("Updated total:", total)



    # for pred, true in zip(predictions, labels):
    #     if(pred - true > -1):
    #         yu[0]+=1
    # 输出更新后的 yu
    # print("Updated yu:", yu)
    # total = sum(yu)
    # print("Updated total:", total)

    ##训练集正确率###################################################
    # 加载训练数据集
    # testing_data = BrainS18Dataset(sets.data_root, sets.excel_file, sets)
    # data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
    # # 读取图像文件名和对应标签（从 Excel 文件）
    # df = pd.read_excel(sets.excel_file)  # 读取 Excel 文件
    # img_names = df.iloc[:, 0].tolist()  # A列为文件名
    # labels = df.iloc[:, 1].tolist()  # B列为标签
    #
    # # 测试模型并获取预测值
    # predictions = run_ce(data_loader, net, img_names, sets, labels)
    #
    # # 计算平均绝对误差
    # errors = [abs(pred - true) for pred, true in zip(predictions, labels)]
    # train_error = np.mean(errors)
    ################################################################
    train_error = 1

    visualize_results(predictions, labels, mean_error,train_error)

    #print(f"Mean Absolute Error (MAE): {mean_error:.2f}")

