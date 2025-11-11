########单个年龄回归
import torch
from torch import nn
from models import resnet

def generate_model(opt):
    """
    根据配置生成模型
    """
    assert opt.model in ['resnet'], "Only 'resnet' model is supported."

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200], "Invalid ResNet depth."

        # 创建指定深度的 ResNet 模型
        if opt.model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                num_seg_classes=1,  # 这里传入num_seg_classes，回归任务可以是1
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda
            )
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                num_seg_classes=1,  # 这里传入num_seg_classes，回归任务可以是1
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda
            )
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                num_seg_classes=1,  # 这里传入num_seg_classes，回归任务可以是1
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda
            )
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                num_seg_classes=1,  # 这里传入num_seg_classes，回归任务可以是1
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda
            )
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                num_seg_classes=1,  # 这里传入num_seg_classes，回归任务可以是1
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda
            )
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                num_seg_classes=1,  # 这里传入num_seg_classes，回归任务可以是1
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda
            )
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                num_seg_classes=1,  # 这里传入num_seg_classes，回归任务可以是1
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda
            )

        # 修改 ResNet 的最后一层
        model.fc = nn.Linear(model.fc.in_features, 1)  # 回归任务：输出单值（年龄）

    # 如果启用了 CUDA，则设置 GPU
    if not opt.no_cuda:
        # if len(opt.gpu_id) > 1:  # 多 GPU
        #     model = model.cuda()
        #     model = nn.DataParallel(model, device_ids=opt.gpu_id)
        # else:  # 单 GPU
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
        print(f"Using GPU: {str(opt.gpu_id[0])}")
        model = model.cuda()
        #model = nn.DataParallel(model, device_ids=None)
    else:
        model = model

    # 加载预训练模型
    if opt.phase != 'test' and opt.pretrain_path:
        print(f"Loading pretrained model from {opt.pretrain_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrain = torch.load(opt.pretrain_path, map_location=device)
        model.load_state_dict(pretrain['state_dict'], strict=False)  # 加载时不强制匹配

    return model, model.parameters()







#
#
#
# #######多任务回归
# import os
# import torch
# from torch import nn
# from models import resnet
#
#
# def generate_model(opt):
#     """
#     根据配置生成模型
#     """
#     assert opt.model in ['resnet'], "Only 'resnet' model is supported."
#     assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200], "Invalid ResNet depth."
#
#     # 创建指定深度的 ResNet 模型
#     if opt.model_depth == 10:
#         model = resnet.resnet10(
#             sample_input_W=opt.input_W,
#             sample_input_H=opt.input_H,
#             sample_input_D=opt.input_D,
#             shortcut_type=opt.resnet_shortcut,
#             no_cuda=opt.no_cuda,
#             num_tasks=4
#         )
#     elif opt.model_depth == 18:
#         model = resnet.resnet18(
#             sample_input_W=opt.input_W,
#             sample_input_H=opt.input_H,
#             sample_input_D=opt.input_D,
#             shortcut_type=opt.resnet_shortcut,
#             no_cuda=opt.no_cuda,
#             num_tasks=4
#         )
#     elif opt.model_depth == 34:
#         model = resnet.resnet34(
#             sample_input_W=opt.input_W,
#             sample_input_H=opt.input_H,
#             sample_input_D=opt.input_D,
#             shortcut_type=opt.resnet_shortcut,
#             no_cuda=opt.no_cuda,
#             num_tasks=4
#         )
#     elif opt.model_depth == 50:
#         model = resnet.resnet50(
#             sample_input_W=opt.input_W,
#             sample_input_H=opt.input_H,
#             sample_input_D=opt.input_D,
#             shortcut_type=opt.resnet_shortcut,
#             no_cuda=opt.no_cuda,
#             num_tasks=4
#         )
#     elif opt.model_depth == 101:
#         model = resnet.resnet101(
#             sample_input_W=opt.input_W,
#             sample_input_H=opt.input_H,
#             sample_input_D=opt.input_D,
#             shortcut_type=opt.resnet_shortcut,
#             no_cuda=opt.no_cuda,
#             num_tasks=4
#         )
#     elif opt.model_depth == 152:
#         model = resnet.resnet152(
#             sample_input_W=opt.input_W,
#             sample_input_H=opt.input_H,
#             sample_input_D=opt.input_D,
#             shortcut_type=opt.resnet_shortcut,
#             no_cuda=opt.no_cuda,
#             num_tasks=4
#         )
#     elif opt.model_depth == 200:
#         model = resnet.resnet200(
#             sample_input_W=opt.input_W,
#             sample_input_H=opt.input_H,
#             sample_input_D=opt.input_D,
#             shortcut_type=opt.resnet_shortcut,
#             no_cuda=opt.no_cuda,
#             num_tasks=4
#         )
#
#     # 如果启用了 CUDA，则设置 GPU
#     if not opt.no_cuda:
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
#         print(f"Using GPU: {str(opt.gpu_id[0])}")
#         model = model.cuda()
#     else:
#         model = model
#
#     # 加载预训练模型
#     if opt.phase != 'test' and opt.pretrain_path:
#         print(f"Loading pretrained model from {opt.pretrain_path}")
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         pretrain = torch.load(opt.pretrain_path, map_location=device)
#
#         # 当加载预训练模型时，如果模型结构不完全匹配，可以忽略不匹配的部分
#         state_dict = pretrain['state_dict']
#         model_state_dict = model.state_dict()
#         for name, param in state_dict.items():
#             if name in model_state_dict:
#                 model_state_dict[name].copy_(param)
#             else:
#                 print(f"Parameter {name} not found in model.")
#
#     return model, model.parameters()