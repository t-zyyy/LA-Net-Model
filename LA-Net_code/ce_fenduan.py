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











####导出excel######第一次

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

def filter_and_predict(predictions, labels, img_names, sets, age_range, model):
    indices = [i for i, pred in enumerate(predictions) if age_range[0] <= pred < age_range[1]]
    filtered_labels = [labels[i] for i in indices]
    filtered_img_names = [img_names[i] for i in indices]

    if not indices:
        print(f"No data in age range {age_range}")
        return [], [], []

    subset_data = torch.utils.data.Subset(BrainS18Dataset(sets.test_root, sets.test_file, sets), indices)
    data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
    filtered_predictions = run_ce(data_loader, model, filtered_img_names, sets, filtered_labels)

    return filtered_predictions, filtered_labels, filtered_img_names

if __name__ == '__main__':
    sets = parse_opts()
    sets.target_type = "age_regression"
    sets.phase = 'test'

    sets.test_path = ''
    sets.input_D = 60
    # sets.test_root = '/home/zsq/train/pre_process/test60'
    # sets.test_file = '/home/zsq/train/pre_process/test.xlsx'
    # sets.test_root = '/home/zsq/train/COPD_60'
    # sets.test_file = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/COPD/COPD男.xlsx'
    sets.test_root = '/disk2/zsq/train/pre_process/test20'
    sets.test_file = '/disk2/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/test偏差.xlsx'
    # sets.test_root = '/home/zsq/train/COPD_60'
    # sets.test_file = '/home/zsq/train/pre_process/DATE/符合分布的train和test/COPD.xlsx'

# ###直接读取数据
#     txt_file_path = "/home/zsq/train/pre_process/DATE/3.25pre.txt"
#     with open(txt_file_path, "r") as file:
#         predictions = [float(line.strip()) for line in file if line.strip()]
#
#     df = pd.read_excel(sets.test_file)
#     img_names = df.iloc[:, 0].tolist()
#     labels = df.iloc[:, 1].tolist()

##经过一下3.25
    checkpoint = torch.load(sets.test_path, map_location=torch.device('cpu'))
    model_state_dict = checkpoint['state_dict']
    new_state_dict = {key.replace('module.', ''): value for key, value in model_state_dict.items()}

    base_model, _ = generate_model(sets)
    base_model.load_state_dict(new_state_dict)

    n = 616
    testing_data = BrainS18Dataset(sets.test_root, sets.test_file, sets)
    subset_data = torch.utils.data.Subset(testing_data, range(n))
    data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)

    df = pd.read_excel(sets.test_file)
    img_names = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()[:n]

    # 测试模型并获取预测值
    predictions = run_ce(data_loader, base_model, img_names, sets, labels)
    errors = [abs(pred - true) for pred, true in zip(predictions, labels)]
    mean_error = np.mean(errors)
    print(mean_error)
    # df.iloc[:, 2] = predictions
    df['C']=predictions
    output_path = r'/disk2/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_20pin/resnet_34/resnet34.xlsx'
    df.to_excel(output_path)

















#
# #####导出excel######第二次
#
# import os
# import torch
# import pandas as pd
# from torch.utils.data import DataLoader
#
# def filter_and_predict(predictions, labels, img_names, sets, age_range, model):
#     indices = [i for i, pred in enumerate(predictions) if age_range[0] <= pred < age_range[1]]
#     filtered_labels = [labels[i] for i in indices]
#     filtered_img_names = [img_names[i] for i in indices]
#
#     if not indices:
#         print(f"No data in age range {age_range}")
#         return [], [], []
#
#     subset_data = torch.utils.data.Subset(BrainS18Dataset(sets.test_root, sets.test_file, sets), indices)
#     data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
#     filtered_predictions = run_ce(data_loader, model, filtered_img_names, sets, filtered_labels)
#
#     return filtered_predictions, filtered_labels, filtered_img_names
#
# if __name__ == '__main__':
#     sets = parse_opts()
#     sets.target_type = "age_regression"
#     sets.phase = 'test'
#     sets.test_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34(zengqiang)_5/resnet_34/epoch_87_batch_1_3.252322.pth.tar'
#     sets.input_D = 60
#     sets.test_root = '/home/zsq/train/pre_process/test60'
#     sets.test_file = '/home/zsq/train/pre_process/test.xlsx'
#
#     txt_file_path = "/home/zsq/train/pre_process/DATE/3.25pre.txt"
#     with open(txt_file_path, "r") as file:
#         predictions = [float(line.strip()) for line in file if line.strip()]
#
#     df = pd.read_excel(sets.test_file)
#     img_names = df.iloc[:, 0].tolist()
#     labels = df.iloc[:, 1].tolist()
#
#     models_paths = [
#         "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_5(35_40)/resnet_34/epoch_187_batch_340_2.077954.pth.tar",
#         "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_4(40_45)/resnet_34/epoch_73_batch_170_3.0116792.pth.tar",
#         "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_7_3(45_50)/resnet_34/epoch_55_batch_1_3.261561.pth.tar",
#         "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_3(50_55)/resnet_34/epoch_32_batch_1_3.200833.pth.tar",
#         "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_1(55_60)/resnet_34/epoch_73_batch_406_3.436912.pth.tar",
#         "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_2(45_74)/resnet_34/epoch_57_batch_1_3.619108.pth.tar"
#     ]
#     age_ranges = [(35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 74)]
#
#     output_dir = "/home/zsq/train/pre_process/DATE/linshi3"
#     os.makedirs(output_dir, exist_ok=True)
#
#     for idx, (model_path, age_range) in enumerate(zip(models_paths, age_ranges)):
#         print(f"Processing model {idx + 1} for age range {age_range}")
#
#         # Load model
#         checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
#         model_state_dict = checkpoint['state_dict']
#         new_state_dict = {key.replace('module.', ''): value for key, value in model_state_dict.items()}
#
#         model, _ = generate_model(sets)
#         model.load_state_dict(new_state_dict)
#
#         # Predict for the specific age range
#         preds, lbls, img_names_for_range = filter_and_predict(predictions, labels, img_names, sets, age_range, model)
#
#         if preds and lbls:
#             # Calculate MAE for the current group
#             model_errors = [abs(pred - true) for pred, true in zip(preds, lbls)]
#             model_mae = sum(model_errors) / len(model_errors)
#             print(f"Model {idx + 1} MAE for age range {age_range}: {model_mae:.2f}")
#         else:
#             print(f"No predictions for model {idx + 1} in age range {age_range}")
#
#         # Export results to Excel
#         output_path = os.path.join(output_dir, f"age_range_{age_range[0]}_{age_range[1]}.xlsx")
#         df_to_save = pd.DataFrame({
#             "File Name": img_names_for_range,
#             "True Age": lbls
#         })
#         df_to_save.to_excel(output_path, index=False)
#         print(f"Data for age range {age_range} saved to {output_path}")
#
#
#
#
#







