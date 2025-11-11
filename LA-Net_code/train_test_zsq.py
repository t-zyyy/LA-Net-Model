###############原始，不加性别标签
'''
Training code for lung CT age regression
Modified by <Your Name>
'''

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import time
from utils.logger import log
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#使系统智能看到编号为1的GPU
import torch
import logging
import glob

import numpy as np
from torch.utils.data import DataLoader, Subset
from setting import parse_opts
from brains import BrainS18Dataset  ###测试集用数据增强
from brain_test import BrainS18 #####验证集不数据增强
from model import generate_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from collections import OrderedDict
import math


# 设置为避免不必要的警告
np.seterr(all='ignore')

# 初始化日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global_num=0
best_mean_error = float('inf')  # 初始化为无穷大
#best_mean_error = 4.89

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

            device = torch.device(f"cuda:{sets.gpu_id[0]}")
            if not sets.no_cuda:
                volume = volume.to(device)

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


################平均绝对误差
def run_inference(model, test_root_path, log_file, output_base_dir):
    """
    对单个模型进行推理，并返回其 mean_error。
    """
    print("Running inference...")

    # 加载设置
    sets = parse_opts()
    sets.target_type = "age_regression"
    sets.phase = 'test'

    testing_data = BrainS18(sets.test_root, sets.test_file, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)

    df = pd.read_excel(sets.test_file)
    img_names = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()
    # labels = labels[:n]

    predictions = run_ce(data_loader, model, img_names, sets, labels)

    errors = [abs(pred - true) for pred, true in zip(predictions, labels)]
    mean_error = np.mean(errors)

    return mean_error


def load_checkpoint(model, optimizer, resume_path, sets):
    # 确定使用的设备
    available_gpus = list(range(torch.cuda.device_count()))

    if not sets.no_cuda and sets.gpu_id[0] in available_gpus:
        device = torch.device(f"cuda:{sets.gpu_id[0]}")
    else:
        print("Specified GPU ID is not available or CUDA is not used. Falling back to CPU.")
        device = torch.device("cpu")

    # 使用 lambda 表达式作为 map_location 参数，确保所有张量都被正确映射到实际可用的设备
    def map_to_device(storage, location):
        return storage.cuda(sets.gpu_id[0]) if not sets.no_cuda and sets.gpu_id[0] in available_gpus else storage.cpu()

    checkpoint = torch.load(resume_path, map_location=map_to_device)

    # 处理 DataParallel 的前缀问题
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    is_data_parallel = any(key.startswith('module.') for key in state_dict.keys())
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] if is_data_parallel and k.startswith('module.') else k  # 移除 'module.' 前缀
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)  # 确保模型在正确的设备上

    # 如果需要恢复优化器状态
    if optimizer is not None and 'optimizer' in checkpoint:
        # 将优化器的状态字典中的张量也映射到正确的设备
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"Checkpoint loaded from {resume_path} on device {device}")

    return checkpoint.get('epoch', 0), checkpoint.get('batch_id', 0)




def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets, save_loss):
    global best_mean_error  # 初始化最佳平均误差为全局变量

    if 'best_mean_error' not in globals():
        best_mean_error = math.inf

    batches_per_epoch = len(data_loader)
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.SmoothL1Loss()  #在这个分模型任务中尝试一下平滑L1损失函数
    # loss_fn = torch.nn.HuberLoss(delta=4.5)
    # loss_fn = torch.nn.L1Loss()  # MAE 损失函数

    # 设备设置
    if not sets.no_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{sets.gpu_id[0]}")
        print(f"Using GPU: {sets.gpu_id[0]}")
        model = model.to(device)
        # loss_fn = loss_fn.to(device)
    else:
        device = torch.device("cpu")
        print("Using CPU")
        model = model.to(device)

    # 加载检查点（如果存在）
    start_epoch, start_batch_id = 0, 0
    if sets.resume_path:
        start_epoch, start_batch_id = load_checkpoint(model, optimizer, sets.resume_path, sets)
        print(f'Resuming training from epoch {start_epoch}, batch {start_batch_id}')

    model.train()
    train_time_sp = time.time()

    # 定义累积梯度的步数
    accumulation_steps = 1  # 这个值可以根据你的显存和需求调整
    for epoch in range(start_epoch, total_epochs):
        running_loss = 0.0
        print(f'Start epoch {epoch}')


        for batch_id, (volumes, labels) in enumerate(tqdm(data_loader, total=batches_per_epoch, desc=f"Epoch {epoch}", ncols=100)):
            if epoch == start_epoch and batch_id < start_batch_id:
                continue

            batch_id_sp = epoch * batches_per_epoch + batch_id

            volumes = volumes.to(device)
            labels = labels.to(device).float()

            # 只在需要时清零梯度（即每累积到指定步数后）
            if (batch_id % accumulation_steps == 0) or (batch_id == 0):
                optimizer.zero_grad()  # 在每个累积周期开始时清零梯度

            predictions = model(volumes)

            # 使用损失函数
            loss = loss_fn(predictions.squeeze(), labels.squeeze())#正常加权函数
            # loss = weighted_da(predictions.squeeze(), labels.squeeze())# 使用偏大/偏小的损失函数
            # loss = weighted_mse_loss(predictions.squeeze(), labels.squeeze())# 使用加权mse损失函数
            # loss = weighted_huber_loss(predictions.squeeze(), labels.squeeze())# 使用加权Huber损失函数

            # 平均损失，因为我们将累积多个批次的梯度
            loss = loss / accumulation_steps
            loss.backward()

            max_norm = 1.0  # 根据实际情况调整这个值
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # 每累积到指定步数后，执行一次权重更新
            if (batch_id + 1) % accumulation_steps == 0 or (batch_id + 1 == len(data_loader)):
                optimizer.step()
                optimizer.zero_grad()  # 更新后重置梯度

            running_loss += loss.item() * accumulation_steps  # 累加原始损失值
            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)

            # 记录当前批次的损失等信息
            print(
                f'Batch: {epoch}-{batch_id} ({batch_id_sp}), loss = {loss.item() * accumulation_steps:.3f}, avg_batch_time = {avg_batch_time:.3f}'
            )

            # 保存模型条件1
            if batch_id_sp % save_interval == 0 and batch_id_sp != 0:
                model.eval()
                with torch.no_grad():
                    current_mean_error = run_inference(model, None, None, './inference_results')
                    #current_mse =     run_inference_mse(model, None, None, './inference_results')
                #print(current_mean_error)
                model.train()
                print(current_mean_error)
                #print(current_mse)
                # 指定文件的完整路径
                output_file_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_60pin/resnet_34/output_error60.txt'

                # 确保目录存在，如果不存在则创建
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                # 打开文件进行追加写入（如果文件不存在，则创建）
                with open(output_file_path, 'a') as f:
                    print(f"Epoch: {epoch}, Batch: {batch_id}, Current Mean Error: {current_mean_error}", file=f)

                if current_mean_error < best_mean_error :
                    best_mean_error = current_mean_error
                    model_save_path = os.path.join(save_folder, f'epoch_{epoch}_batch_{batch_id}_{best_mean_error:.6f}.pth.tar')
                    os.makedirs(save_folder, exist_ok=True)

                    print(f"Best mean error updated to {best_mean_error}")
                    print(f"Saving checkpoint at {model_save_path}")

                    torch.save({
                        'epoch': epoch,
                        'batch_id': batch_id,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, model_save_path)

            # 保存模型条件2
            if batch_id == 1:
            #if batch_id % 100== 0:
                    model.eval()
                    with torch.no_grad():
                        current_mean_error = run_inference(model, None, None, './inference_results')
                        #current_mse = run_inference_mse(model, None, None, './inference_results')
                        # print(current_mean_error)
                    model.train()
                    print(current_mean_error)
                    #print(current_mse)
                    # 指定文件的完整路径
                    output_file_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_60pin/resnet_34/output_error60.txt'

                    # 确保目录存在，如果不存在则创建
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                    # 打开文件进行追加写入（如果文件不存在，则创建）
                    with open(output_file_path, 'a') as f:
                        print(f"Epoch: {epoch}, Batch: {batch_id}, Current Mean Error: {current_mean_error}",
                                file=f)
                    if current_mean_error < best_mean_error:
                        best_mean_error = current_mean_error

                    if 1:

                        model_save_path = os.path.join(save_folder,
                                                           f'epoch_{epoch}_batch_{batch_id}_{best_mean_error:.6f}.pth.tar')
                        os.makedirs(save_folder, exist_ok=True)

                        print(f"Best mean error updated to {best_mean_error}")
                        print(f"Saving checkpoint at {model_save_path}")

                        torch.save({
                            'epoch': epoch,
                            'batch_id': batch_id,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                        }, model_save_path)

        scheduler.step()
        print(f'Learning rate after epoch {epoch} = {scheduler.get_last_lr()}')

        # 打印每个epoch结束时的平均损失
        avg_loss = running_loss / batches_per_epoch
        print(f'Epoch {epoch} finished, average loss: {avg_loss:.3f}')

        avg_epoch_loss = running_loss / batches_per_epoch
        print(f'End of epoch {epoch}, average loss = {avg_epoch_loss:.3f}')

    # if batch_id == 2222:
    #     model_save_path = os.path.join(save_folder, f'epoch_{epoch}_batch_{batch_id}_{current_mean_error:.6f}.pth.tar')
    #     os.makedirs(save_folder, exist_ok=True)
    #
    #     print(f"Best mean error updated to {best_mean_error}")
    #     print(f"Saving checkpoint at {model_save_path}")
    #
    #     torch.save({
    #         'epoch': epoch,
    #         'batch_id': batch_id,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict()
    #     }, model_save_path)


    print('Training finished!')

if __name__ == '__main__':
    # 设置
    sets = parse_opts()
    device = torch.device("cuda" if torch.cuda.is_available() and not sets.no_cuda else 'cpu')
    sets.ci_test = False
    sets.no_cuda = False

    if sets.ci_test:  # CI测试参数
        sets.excel_file = './toy_data/test_ci.txt'
        sets.n_epochs = 1
        sets.no_cuda = True
        sets.data_root = './toy_data'
        sets.pretrain_path = ''
        sets.num_workers = 0

        sets.model_depth = 10
        sets.resnet_shortcut = 'A'
        sets.input_D = 14
        sets.input_H = 28
        sets.input_W = 28

    # 加载模型
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)
    print(model)

    # 定义优化器和调度器
    params = [{'params': parameters, 'lr': sets.learning_rate}]
    optimizer = torch.optim.SGD(params, momentum=0.96, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 从检查点恢复
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print(f"=> loading checkpoint '{sets.resume_path}'")
            # 加载检查点
            checkpoint = torch.load(sets.resume_path, map_location=torch.device(f'cuda:{sets.gpu_id[0]}'))
            #checkpoint = torch.load(sets.resume_path, map_location=lambda storage, loc: storage.cuda(0))
            # 获取模型的 state_dict
            new_state_dict = checkpoint['state_dict']  # 使用 'state_dict' 键来获取保存的模型状态

            # 如果模型是使用 DataParallel 保存的，去掉 "module." 前缀
            #new_state_dict = {k.replace('module.', ''): v for k, v in new_state_dict.items()}

            # 加载模型的 state_dict
            model.load_state_dict(new_state_dict, strict=False)

            # 恢复优化器的状态
            optimizer.load_state_dict(checkpoint['optimizer'])

            # 恢复 epoch 信息
            epoch = checkpoint['epoch']

            print(f"=> loaded checkpoint '{sets.resume_path}' (epoch {epoch})")
        else:
            print(f"=> no checkpoint found at '{sets.resume_path}'")


    # 数据加载
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True

    training_dataset = BrainS18Dataset(sets.data_root, sets.excel_file, sets)
    data_loader = DataLoader(
        training_dataset,
        batch_size=sets.batch_size,
        shuffle=True,
        num_workers=sets.num_workers,
        pin_memory=sets.pin_memory,
    )

    # 将模型移到选择的设备（GPU 或 CPU）
    model.to(device)

    # 开始训练
    train(
        data_loader,
        model,
        optimizer,
        scheduler,
        total_epochs=sets.n_epochs,
        save_interval=sets.save_intervals,
        save_folder=sets.save_folder,
        sets=sets,
        save_loss=sets.save_loss
    )






