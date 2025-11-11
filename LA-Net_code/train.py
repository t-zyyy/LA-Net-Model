'''
Training code for lung CT age regression
Modified by <Your Name>
'''

from tqdm import tqdm
from setting import parse_opts
from brains import BrainS18Dataset
from model import generate_model
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import time
from utils.logger import log
import os
import openpyxl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"#使系统智能看到编号为3的GPU
import torch

global_num=0

def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets ,save_loss):
    global global_num
    """
    训练函数
    """
    batches_per_epoch = len(data_loader)  # 每个epoch的批次数量
    log.info(f'{total_epochs} epochs in total, {batches_per_epoch} batches per epoch')
    loss_fn = nn.MSELoss()  # 定义均方误差损失函数

    if not sets.no_cuda:  # 如果使用GPU
        loss_fn = loss_fn.cuda()

    model.train()  # 设置模型为训练模式
    train_time_sp = time.time()  # 记录训练开始时间

    for epoch in range(total_epochs):  # 循环每个epoch
        log.info(f'Start epoch {epoch}')
        scheduler.step()  # 更新学习率
        log.info(f'Learning rate = {scheduler.get_last_lr()}')

        for batch_id, (volumes, labels) in tqdm(enumerate(data_loader), total=batches_per_epoch, desc=f"Epoch {epoch}", ncols=100):
            batch_id_sp = epoch * batches_per_epoch + batch_id  # 当前批次的全局ID
            if not sets.no_cuda:
                volumes = volumes.to(device)  # 将数据迁移到GPU
                labels = labels.to(device).float()  # 将标签迁移到GPU并转换为float
                #print(labels.dtype)  # 打印输入的数据类型

            optimizer.zero_grad()  # 清除之前的梯度
            predictions = model(volumes)  # 前向传播，得到预测结果

            loss = loss_fn(predictions.squeeze(), labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)  # 计算平均每批耗时
            log.info(
                f'Batch: {epoch}-{batch_id} ({batch_id_sp}), loss = {loss.item():.3f}, avg_batch_time = {avg_batch_time:.3f}'
            )
            if loss<save_loss:
                global_num+= global_num

            # 保存模型
            if (batch_id_sp % save_interval == 0 and batch_id_sp != 0 ) or (loss<save_loss and global_num>=10):
                global_num=0
                model_save_path = os.path.join(save_folder, f'epoch_{epoch}_batch_{batch_id}.pth.tar')
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                log.info(f'Saving checkpoint: epoch = {epoch}, batch_id = {batch_id}')
                torch.save({
                    'epoch': epoch,
                    'batch_id': batch_id,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, model_save_path)

    print('Training finished!')



if __name__ == '__main__':

    # 设置
    sets = parse_opts()
    device = torch.device("cuda"  if torch.cuda.is_available() else 'cpu')
    sets.ci_test = False
    sets.no_cuda = False
    #sets.gpu_id = [1]  # 默认使用第 1 个 GPU

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

    params = [{'params': parameters, 'lr': sets.learning_rate}]
    optimizer = optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # 从检查点恢复
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print(f"=> loading checkpoint '{sets.resume_path}'")
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{sets.resume_path}' (epoch {checkpoint['epoch']})")

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
        #worker_init_fn = worker_init_fn  # 为每个 worker 设置设备
    )

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
