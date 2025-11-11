import numpy as np
from torch.utils.data import DataLoader
from setting import parse_opts
from brains import BrainS18Dataset
from model import generate_model
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

np.seterr(all='ignore')

def run_ce(data_loader, model, img_names, sets, labels):
    predictions = []
    model.eval()

    with tqdm(data_loader, desc="Processing Samples", unit="batch") as pbar:
        for batch_id, batch_data in enumerate(pbar):
            volume, _ = batch_data

            if isinstance(volume, (list, np.ndarray)):
                volume = torch.tensor(volume, dtype=torch.float32)

            if not isinstance(volume, torch.Tensor):
                raise TypeError(f"Expected volume to be a Tensor, but got {type(volume)}")

            if not sets.no_cuda:
                volume = volume.cuda()

            with torch.no_grad():
                pred = model(volume)
                pred = pred.cpu().numpy().flatten()
                predictions.append(pred[0])

            tqdm.write(f"Batch {batch_id + 1}: Predicted Age = {pred[0]:.2f}, True Age = {labels[batch_id]:.2f}")

    return predictions


def visualize_results(predictions, labels, mean_error):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(labels, predictions, color='blue', alpha=0.6)
    plt.plot([min(labels), max(labels)], [min(labels), max(labels)], color='red', linestyle='--')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title(f'True vs Predicted Age\nMAE: {mean_error:.2f}')

    errors = [abs(pred - true) for pred, true in zip(predictions, labels)]
    n = len(errors)
    weights = np.ones_like(errors) / n * 100

    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=20, weights=weights, color='green', alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency (%)')
    plt.title(f'Error Distribution\nMAE: {mean_error:.2f}')

    plt.tight_layout()
    plt.show()


def filter_and_predict(predictions, labels, img_names, sets, age_range, model):
    indices = [i for i, pred in enumerate(predictions) if age_range[0] <= pred < age_range[1]]
    filtered_labels = [labels[i] for i in indices]
    filtered_img_names = [img_names[i] for i in indices]

    if not indices:
        print(f"No data in age range {age_range}")
        return [], []

    subset_data = torch.utils.data.Subset(BrainS18Dataset(sets.test_root, sets.test_file, sets), indices)
    data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
    filtered_predictions = run_ce(data_loader, model, filtered_img_names, sets, filtered_labels)

    return filtered_predictions, filtered_labels


def save_predictions_to_excel(img_names, predictions, predictions2, output_path):
    # 对 predictions2 进行填充，以确保长度与 img_names 一致
    predictions2 = predictions2 + [None] * (len(img_names) - len(predictions2))

    # 将结果保存为一个 DataFrame
    data = {
        'File Name': img_names,
        'First Prediction': predictions,
        'Second Prediction': predictions2
    }

    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    print(f"Predictions saved to {output_path}")


# 计算分段 MAE 和总 MAE
def main():
    sets = parse_opts()
    sets.target_type = "age_regression"
    sets.phase = 'test'
    sets.test_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34(zengqiang)_5/resnet_34/epoch_87_batch_1_3.252322.pth.tar'
    sets.input_D = 60
    sets.test_root = '/home/zsq/train/COPD_60'
    sets.test_file = '/home/zsq/train/pre_process/DATE/COPD_new/轻度_一级.xlsx'

    checkpoint = torch.load(sets.test_path, map_location=torch.device('cpu'))
    model_state_dict = checkpoint['state_dict']
    new_state_dict = {key.replace('module.', ''): value for key, value in model_state_dict.items()}

    base_model, _ = generate_model(sets)
    base_model.load_state_dict(new_state_dict)

    n = 329  # 这里可以调整测试样本数量
    testing_data = BrainS18Dataset(sets.test_root, sets.test_file, sets)
    subset_data = torch.utils.data.Subset(testing_data, range(n))
    data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)

    df = pd.read_excel(sets.test_file)
    img_names = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()[:n]

    # 测试基础模型并获取预测值
    predictions = run_ce(data_loader, base_model, img_names, sets, labels)
    errors = [abs(pred - true) for pred, true in zip(predictions, labels)]
    mean_error = np.mean(errors)
    print(f"Mean Error from base model: {mean_error:.2f}")

    # 定义各个分段模型路径
    models_paths = [
        "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34(zengqiang)_5/resnet_34/epoch_87_batch_1_3.252322.pth.tar",
        # 2.18
        "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_4(40_45)/resnet_34/epoch_73_batch_170_3.0116792.pth.tar",
        # 3.12
        "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_7_3(45_50)/resnet_34/epoch_55_batch_1_3.261561.pth.tar",
        # 3.39
        "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_3(50_55)/resnet_34/epoch_32_batch_1_3.200833.pth.tar",
        # 3.36
        "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_1(55_60)/resnet_34/epoch_73_batch_406_3.436912.pth.tar",
        # 3.62
        "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_2(45_74)/resnet_34/epoch_57_batch_1_3.619108.pth.tar"
        # 3.71
    ]
    age_ranges = [(0, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 100)]

    final_predictions2 = []  # 保存六个模型的分段预测结果

    for model_path, age_range in zip(models_paths, age_ranges):
        print(f"Processing model for age range {age_range}")

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_state_dict = checkpoint['state_dict']
        new_state_dict = {key.replace('module.', ''): value for key, value in model_state_dict.items()}

        model, _ = generate_model(sets)
        model.load_state_dict(new_state_dict)

        preds, _ = filter_and_predict(predictions, labels, img_names, sets, age_range, model)
        final_predictions2.extend(preds)

    # 填充第二次预测结果，确保与文件名数量一致
    final_predictions2 = final_predictions2 + [None] * (len(img_names) - len(final_predictions2))

    # 检查预测结果长度是否匹配
    if len(predictions) != len(final_predictions2):
        print(f"警告: 两次预测结果长度不匹配：{len(predictions)} vs {len(final_predictions2)}")

    # 保存到Excel
    output_excel_path = '/home/zsq/train/pre_process/DATE/COPD_new/分析/轻度.xlsx'  # 请更改为实际路径
    save_predictions_to_excel(img_names[:n], predictions, final_predictions2, output_excel_path)


if __name__ == '__main__':
    main()
