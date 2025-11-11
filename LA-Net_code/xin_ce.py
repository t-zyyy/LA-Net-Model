# import numpy as np
# from torch.utils.data import DataLoader, Subset
# from setting import parse_opts
# from brains import BrainS18Dataset
# from model import generate_model
# import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm import tqdm  # 导入 tqdm 库来显示进度条
# import glob
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使系统智能看到编号为3的GPU
# import torch
#
#
# # 设置为避免不必要的警告
# np.seterr(all='ignore')
#
#
# def run_ce(data_loader, model, img_names, sets, labels):
#     """
#     测试回归模型，并计算预测值与真实值的差异。
#     """
#     predictions = []
#     model.eval()  # 切换到评估模式
#
#     with tqdm(data_loader, desc="Processing Samples", unit="batch") as pbar:
#         for batch_id, batch_data in enumerate(pbar):
#             volume, _ = batch_data  # 假设 batch_data 包含图像和标签
#
#             if isinstance(volume, list) or isinstance(volume, np.ndarray):
#                 volume = torch.tensor(volume, dtype=torch.float32)  # 转换为 Tensor
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
#
#             tqdm.write(f"Batch {batch_id + 1}: Predicted Age = {pred[0]:.2f}, True Age = {labels[batch_id]:.2f}")
#
#     for idx, pred in enumerate(predictions):
#         print(f"Image: {img_names[idx]} | Predicted Age: {pred:.2f} | True Age: {labels[idx]}")
#
#     return predictions
#
#
# def visualize_results(predictions, labels, mean_error, train_error):
#     """
#     可视化预测值与真实值的对比，并绘制误差分布。
#     """
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.scatter(labels, predictions, color='blue', alpha=0.6)
#     plt.plot([min(labels), max(labels)], [min(labels), max(labels)], color='red', linestyle='--')
#     plt.xlabel('True Age')
#     plt.ylabel('Predicted Age')
#     plt.title(f'True vs Predicted Age\ntrain Absolute Error (MAE): {train_error:.2f}')
#
#     errors = [abs(pred - true) for pred, true in zip(predictions, labels)]
#     n = len(errors)
#     weights = np.ones_like(errors) / n * 100
#
#     plt.subplot(1, 2, 2)
#     plt.hist(errors, bins=20, weights=weights, color='green', alpha=0.7)
#     plt.xlabel('Prediction Error')
#     plt.ylabel('Frequency (%)')
#     plt.title(f'Error Distribution\nval Absolute Error (MAE): {mean_error:.2f}')
#
#     plt.tight_layout()
#     plt.show()
#
#
# def run_inference_for_all_models(test_root_path):
#     model_paths = glob.glob(os.path.join(test_root_path, "*.pth.tar"))
#
#     for model_path in model_paths:
#         print(f"Loading and testing model: {model_path}")
#
#         # 加载设置
#         sets = parse_opts()
#         sets.target_type = "age_regression"
#         sets.phase = 'test'
#         sets.test_path = model_path
#
#         checkpoint = torch.load(sets.test_path, map_location=torch.device('cpu'))
#         # 如果模型是使用 DataParallel 保存的，去除 'module.' 前缀
#         model_state_dict = checkpoint['state_dict']  # 假设模型权重在 'state_dict' 中
#         new_state_dict = {}
#
#         for key, value in model_state_dict.items():
#             new_key = key.replace('module.', '')  # 去掉 'module.' 前缀
#             new_state_dict[new_key] = value
#
#         net, _ = generate_model(sets)
#
#         # 检查模型是否已经是 DataParallel 包装过的
#         if isinstance(net, torch.nn.DataParallel):
#             net = net.module
#
#         # 加载模型权重
#         net.load_state_dict(new_state_dict)
#
#         # if not isinstance(net, torch.nn.DataParallel):
#         #     net = torch.nn.DataParallel(net)
#         #
#         # net.load_state_dict(new_state_dict)
#
#         n = 275
#         testing_data = BrainS18Dataset(sets.test_root, sets.test_file, sets)
#         subset_data = Subset(testing_data, range(n))
#         data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
#
#         df = pd.read_excel(sets.test_file)
#         img_names = df.iloc[:, 0].tolist()
#         labels = df.iloc[:, 1].tolist()
#         labels = labels[:n]
#
#         predictions = run_ce(data_loader, net, img_names, sets, labels)
#
#         errors = [abs(pred - true) for pred, true in zip(predictions, labels)]
#         mean_error = np.mean(errors)
#
#         ##训练集正确率###################################################
#         training_data = BrainS18Dataset(sets.data_root, sets.excel_file, sets)
#         train_loader = DataLoader(training_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
#         df_train = pd.read_excel(sets.excel_file)
#         img_names_train = df_train.iloc[:, 0].tolist()
#         labels_train = df_train.iloc[:, 1].tolist()
#
#         predictions_train = run_ce(train_loader, net, img_names_train, sets, labels_train)
#
#         errors_train = [abs(pred - true) for pred, true in zip(predictions_train, labels_train)]
#         train_error = np.mean(errors_train)
#         ################################################################
#
#         visualize_results(predictions, labels, mean_error, train_error)
#
#
# if __name__ == '__main__':
#     test_root_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34/'  # 修改为你的开始路径
#     run_inference_for_all_models(test_root_path)


import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from setting import parse_opts
from brains import BrainS18Dataset
from model import generate_model
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入 tqdm 库来显示进度条
import logging

# 设置为避免不必要的警告
np.seterr(all='ignore')

# 初始化日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_ce(data_loader, model, img_names, sets, labels):
    """
    测试回归模型，并计算预测值与真实值的差异。
    """
    predictions = []
    model.eval()  # 切换到评估模式

    with tqdm(data_loader, desc="Processing Samples", unit="batch") as pbar:
        for batch_id, batch_data in enumerate(pbar):
            volume, _ = batch_data  # 假设 batch_data 包含图像和标签

            if isinstance(volume, list) or isinstance(volume, np.ndarray):
                volume = torch.tensor(volume, dtype=torch.float32)  # 转换为 Tensor

            if not isinstance(volume, torch.Tensor):
                raise TypeError(f"Expected volume to be a Tensor, but got {type(volume)}")

            if not sets.no_cuda:
                volume = volume.cuda()

            with torch.no_grad():
                pred = model(volume)
                pred = pred.cpu().numpy().flatten()
                predictions.append(pred[0])

            tqdm.write(f"Batch {batch_id + 1}: Predicted Age = {pred[0]:.2f}, True Age = {labels[batch_id]:.2f}")

    for idx, pred in enumerate(predictions):
        print(f"Image: {img_names[idx]} | Predicted Age: {pred:.2f} | True Age: {labels[idx]}")

    return predictions


def visualize_results(predictions, labels, mean_error, train_error, output_dir, model_name, log_file):
    """
    可视化预测值与真实值的对比，并绘制误差分布。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(labels, predictions, color='blue', alpha=0.6)
    plt.plot([min(labels), max(labels)], [min(labels), max(labels)], color='red', linestyle='--')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title(f'True vs Predicted Age\ntrain Absolute Error (MAE): {train_error:.2f}')

    errors = [abs(pred - true) for pred, true in zip(predictions, labels)]
    n = len(errors)
    weights = np.ones_like(errors) / n * 100

    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=20, weights=weights, color='green', alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency (%)')
    plt.title(f'Error Distribution\nval Absolute Error (MAE): {mean_error:.2f}')

    plt.tight_layout()

    # 保存图片到指定目录
    output_path = os.path.join(output_dir, f"{model_name}_results.png")
    plt.savefig(output_path)
    #plt.close()  # 关闭图形以释放内存
    print(f"Saved visualization to {output_path}")

    # 将文件名写入日志文件
    log_file.write(f"{output_path}\n")
    log_file.flush()  # 确保立即写入磁盘


def run_inference_for_all_models(test_root_path, log_filename='generated_images.txt'):
    model_paths = glob.glob(os.path.join(test_root_path, "*.pth.tar"))
    print(f"Total number of models found: {len(model_paths)}")

    output_base_dir = './inference_results'
    os.makedirs(output_base_dir, exist_ok=True)

    # 打开日志文件
    log_file_path = os.path.join(output_base_dir, log_filename)
    with open(log_file_path, 'w') as log_file:
        for idx, model_path in enumerate(model_paths, start=1):
            print(f"\n[{idx}/{len(model_paths)}] Loading and testing model: {model_path}")

            # 加载设置
            sets = parse_opts()
            sets.target_type = "age_regression"
            sets.phase = 'test'
            sets.test_path = model_path

            checkpoint = torch.load(sets.test_path, map_location=torch.device('cpu'))
            # 如果模型是使用 DataParallel 保存的，去除 'module.' 前缀
            model_state_dict = checkpoint['state_dict']  # 假设模型权重在 'state_dict' 中
            new_state_dict = {}

            for key, value in model_state_dict.items():
                new_key = key.replace('module.', '')  # 去掉 'module.' 前缀
                new_state_dict[new_key] = value

            net, _ = generate_model(sets)

            # 检查模型是否已经是 DataParallel 包装过的
            if isinstance(net, torch.nn.DataParallel):
                net = net.module

            # 加载模型权重
            net.load_state_dict(new_state_dict)

            n = 377
            testing_data = BrainS18Dataset(sets.test_root, sets.test_file, sets)
            subset_data = Subset(testing_data, range(n))
            data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=12, pin_memory=False)

            df = pd.read_excel(sets.test_file)
            img_names = df.iloc[:, 0].tolist()
            labels = df.iloc[:, 1].tolist()
            labels = labels[:n]

            predictions = run_ce(data_loader, net, img_names, sets, labels)

            errors = [abs(pred - true) for pred, true in zip(predictions, labels)]
            mean_error = np.mean(errors)

            ##训练集正确率###################################################
            training_data = BrainS18Dataset(sets.data_root, sets.excel_file, sets)

            train_loader = DataLoader(training_data, batch_size=1, shuffle=False, num_workers=12, pin_memory=False)
            df_train = pd.read_excel(sets.excel_file)
            img_names_train = df_train.iloc[:, 0].tolist()
            labels_train = df_train.iloc[:, 1].tolist()

            predictions_train = run_ce(train_loader, net, img_names_train, sets, labels_train)

            errors_train = [abs(pred - true) for pred, true in zip(predictions_train, labels_train)]
            train_error = np.mean(errors_train)

            ################################################################

            model_name = os.path.splitext(os.path.basename(model_path))[0]
            model_output_dir = os.path.join(output_base_dir, model_name)

            visualize_results(predictions, labels, mean_error, train_error, model_output_dir, model_name, log_file)
            print(f"[{idx}/{len(model_paths)}] Finished processing model: {model_path}")


if __name__ == '__main__':
    test_root_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_10_3/resnet_34'  # 修改为你的开始路径
    run_inference_for_all_models(test_root_path)