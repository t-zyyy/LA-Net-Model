'''
Configs for training & testing
Modified for age regression task
'''

import argparse



def parse_opts():
    parser = argparse.ArgumentParser()
    # 数据路径
    parser.add_argument(
        '--data_root',
        default='/home/zsq/train/pre_process/train_纯肺_60',       #/home/zsq/train/pre_process/train60
        type=str,
        help='Root directory path of data')
    # Excel 文件路径
    parser.add_argument(
        '--excel_file',
        default='/home/zsq/train/pre_process/DATE/符合分布的train和test/train.xlsx',#
        type=str,
        help='Path for Excel file containing filenames and age labels')
    #test,数据路径
    parser.add_argument(
        '--test_root',
        default='/home/zsq/train/pre_process/test_纯肺_60',#/home/zsq/train/pre_process/test60-1
        type=str,
        help='Root directory path of data')
    # test Excel 文件路径
    parser.add_argument(
        '--test_file',
        default='/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/test/test最新偏差.xlsx',
        #/home/zsq/train/pre_process/DATE/再次做印证模型选择/test_包含所有年龄.xlsx
        type=str,
        help='Path for Excel file containing filenames and age labels')
    # 学习率设置
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.001,#0.00033775440089890197,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    # 工作线程数量
    parser.add_argument(
        '--num_workers',
        default=6,
        type=int,
        help='Number of jobs')
    # 批量大小
    parser.add_argument(
        '--batch_size', default=12, type=int, help='Batch Size')
    # 训练阶段
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    # 保存间隔
    parser.add_argument(
        '--save_intervals',
        default=800,
        type=int,
        help='Interval for saving the model')
    # 保存条件（和上一个是0r的关系）
    parser.add_argument(
        '--save_loss',
        default=8,
        type=int,
        help='loss condition for saving the model')
    # 训练轮次
    parser.add_argument(
        '--n_epochs',
        default=100,
        type=int,
        help='Number of total epochs to run')
    # 输入大小设置
    parser.add_argument(
        '--input_D',
        default=20,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=224,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=224,
        type=int,
        help='Input size of width')
    # 恢复模型路径
    parser.add_argument(
        '--resume_path',
        default='' ,#'./MedicalNet_pytorch_files/trails/models/resnet_10/epoch_2_batch_0.pth.tar'
        type=str,
        help='Path to resume the model.')
    # 恢复模型路径test
    parser.add_argument(###########MO
        '--test_path',
        default='' ,
        type=str,
        help='Path to resume the model.')
    # 预训练模型路径
    parser.add_argument(
        '--pretrain_path',
        default='/home/zsq/train/MedicalNet_pytorch_files/pretrain/resnet_18_23dataset.pth',
        type=str,
        help='Path for pretrained model.')
    # 新层名称
    parser.add_argument(
        '--new_layer_names',
        default=['fc'],  # 回归任务仅需全连接层
        type=list,
        help='New layers to be trained apart from the backbone')
    # CUDA 使用设置
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    # GPU ID 列表
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,
        default=[0],  # 默认 GPU ID 列表为 [0]
        help='Gpu id lists')
    # 模型类型与深度
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='Model type (e.g., resnet)')
    parser.add_argument(
        '--model_depth',
        default=34,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    # ResNet 快捷连接类型
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    # 随机种子
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    # CI 测试
    parser.add_argument(
        '--ci_test', action='store_true', help='If true, CI testing is used.')

    # 输出保存文件夹路径
    args = parser.parse_args()
    args.save_folder = "./MedicalNet_pytorch_files/trails/models/纯肺/60/{}_{}".format(
        args.model, args.model_depth)

    return args
