import numpy as np
from torch.utils.data import DataLoader
from setting import parse_opts#######
from brains import BrainS18Dataset######
from model import generate_model########
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

np.seterr(all='ignore')

def run_ce(data_loader, model, img_names, sets, labels):
    predictions = []
    matched_img_names = []
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
                matched_img_names.append(img_names[batch_id])

            tqdm.write(f"Batch {batch_id + 1}: Predicted Age = {pred[0]:.2f}, True Age = {float(labels[batch_id]):.2f}")

    return matched_img_names, predictions


def filter_and_predict(predictions, labels, img_names, sets, age_range, model):
    indices = [i for i, pred in enumerate(predictions) if age_range[0] <= pred < age_range[1]]
    filtered_labels = [labels[i] for i in indices]
    filtered_img_names = [img_names[i] for i in indices]

    if not indices:
        print(f"No data in age range {age_range}")
        return [], [], []

    subset_data = torch.utils.data.Subset(BrainS18Dataset(sets.test_root, sets.test_file, sets), indices)
    data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)

    matched_img_names, filtered_predictions = run_ce(data_loader, model, filtered_img_names, sets, filtered_labels)

    deviations = [pred - true for pred, true in zip(filtered_predictions, filtered_labels)]
    return matched_img_names, filtered_predictions, deviations


def save_predictions_to_excel(img_names, final_predictions, deviations, output_path):
    data = {
        'File Name': img_names,
        'Final Predicted Value': final_predictions,
        'Prediction Deviation': deviations
    }
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    print(f"Predictions saved to {output_path}")




##/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34/epoch_58_batch_552_3.819779.pth.tar
##0.4
##/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34/epoch_70_batch_250_4.154421329498291.pth.tar
##

def main():
    sets = parse_opts()
    sets.target_type = "age_regression"
    sets.phase = 'test'
    sets.test_path = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_10_1/protect/epoch_86_batch_1_3.545662.pth.tar'
    sets.input_D = 60
    # sets.test_root = '/home/zsq/train/COPD_60'
    # sets.test_file = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/COPD/COPD偏差.xlsx'
    # sets.test_file = '/home/zsq/train/pre_process/DATE/COPD_new/分析偏大偏小/三级+四级.xlsx'
    # sets.test_root = '/home/zsq/train/pre_process/test60'
    # sets.test_file = '/home/zsq/train/pre_process/DATE/符合分布的train和test/test——yuan.xlsx'
    # sets.test_root = '/home/zsq/train/pre_process/PRI60'
    # sets.test_file = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/PRISM/PRISM.xlsx'
    # sets.test_root = '/home/zsq/train/external_shengjing/buchong60'
    # sets.test_file = '/home/zsq/train/pre_process/DATE/外部验证excel/盛京/前后都健康/前后两次无COPD验证.xlsx'
    sets.test_root = '/home/zsq/train/pre_process/PRI60'
    sets.test_file = '/home/zsq/train/pre_process/DATE/外部验证excel/PRISM.xlsx'

    # # 加载主模型并注入 M3d-CAM##########
    # base_model, _ = generate_model(sets)
    # base_model.load_state_dict(new_state_dict)
    # base_model = medcam.inject(  # 新增
    #     base_model,
    #     output_dir="base_model_attention",
    #     save_maps=True,
    #     backend="gcam",
    #     layer="auto"
    # )
########################################3

    #经过一下3.25
    checkpoint = torch.load(sets.test_path, map_location=torch.device('cpu'))
    model_state_dict = checkpoint['state_dict']
    new_state_dict = {key.replace('module.', ''): value for key, value in model_state_dict.items()}

    base_model, _ = generate_model(sets)
    base_model.load_state_dict(new_state_dict)

    n = 546
    testing_data = BrainS18Dataset(sets.test_root, sets.test_file, sets)
    subset_data = torch.utils.data.Subset(testing_data, range(n))
    data_loader = DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)

    df = pd.read_excel(sets.test_file)
    img_names = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()[:n]

    # **执行第一阶段预测**
    matched_img_names, predictions = run_ce(data_loader, base_model, img_names, sets, labels)

    first_deviations = [pred - true for pred, true in zip(predictions, labels)]
    errors = [abs(pred - true) for pred, true in zip(predictions, labels)]
    mean_error = np.mean(errors)
    print(f"Mean Error for first model: {mean_error}")


    # models_paths = [
    #     "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_5(35_40)/resnet_34/epoch_187_batch_340_2.077954.pth.tar",
    #     "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_4(40_45)/resnet_34/epoch_73_batch_170_3.0116792.pth.tar",
    #     "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_7_3(45_50)/resnet_34/epoch_55_batch_1_3.261561.pth.tar",
    #     "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_3(50_55)/resnet_34/epoch_32_batch_1_3.200833.pth.tar",
    #     "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_1(55_60)/resnet_34/epoch_73_batch_406_3.436912.pth.tar",
    #     "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_8_2(45_74)/resnet_34/epoch_57_batch_1_3.619108.pth.tar"
    # ]

    models_paths = [
        "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_10_2/protect/epoch_32_batch_1_3.801963.pth.tar",
        "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_10_1/protect/epoch_86_batch_1_3.545662.pth.tar",
        "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_10_4/protect/epoch_63_batch_1_3.885812.pth.tar"
    ]
    # models_paths = [
    #     "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_10_1/protect/epoch_86_batch_1_3.545662.pth.tar",
    #     "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_10_1/protect/epoch_86_batch_1_3.545662.pth.tar",
    #     "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_10_1/protect/epoch_86_batch_1_3.545662.pth.tar"
    # ]
    age_ranges = [(0, 49), (49, 60), (60, 100)]

    all_results = []

    for model_path, age_range in zip(models_paths, age_ranges):
        print(f"Processing model for age range {age_range}")

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_state_dict = checkpoint['state_dict']
        new_state_dict = {key.replace('module.', ''): value for key, value in model_state_dict.items()}

        model, _ = generate_model(sets)
        model.load_state_dict(new_state_dict)

##############################################3
        # model = medcam.inject(  # 新增
        #     model,
        #     output_dir=f"submodel_{age_range[0]}-{age_range[1]}_attention",
        #     save_maps=True,
        #     backend="gcam",
        #     layer="auto"
        # )


 ############## ...后续推理流程...#################


        filtered_img_names, preds, deviations = filter_and_predict(predictions, labels, matched_img_names, sets, age_range, model)

        all_results.extend(zip(filtered_img_names, preds, deviations))

    all_results.sort(key=lambda x: img_names.index(x[0]))

    sorted_img_names, sorted_final_predictions, sorted_deviations = zip(*all_results) if all_results else ([], [], [])

    output_excel_path = '/home/zsq/train/pre_process/DATE/外部验证excel/PRISM偏差.xlsx'
    save_predictions_to_excel(sorted_img_names, sorted_final_predictions, sorted_deviations, output_excel_path)


if __name__ == '__main__':
    main()
