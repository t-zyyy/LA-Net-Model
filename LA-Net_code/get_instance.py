import os
import numpy as np
import SimpleITK as sitk
from tqdm import trange
import random
import math
from skimage import transform

def get_dir_ct(main_path):
    ctpath = []
    ct_list = os.listdir(main_path)  # 列出文件夹下所有的目录与文件
    for ii in range(0, len(ct_list)):
        path = os.path.join(main_path, ct_list[ii])
        ctpath.append(path)
    return ctpath

def get_instances(ct_path, K, save_path):
    sitk_img = sitk.ReadImage(ct_path)
    img_arr = sitk.GetArrayFromImage(sitk_img)
    slice_num = img_arr.shape[0]

    # 设置随机数生成的范围，排除边缘切片
    min_val = 0
    max_val = slice_num
    cut_off = 0.05 * (max_val - min_val)
    new_min = min_val + cut_off
    new_max = max_val - cut_off

    # 将新区间平均分成 K 份
    step = (new_max - new_min) / K
    select_slice_num = []
    for i in range(K):
        start = new_min + i * step
        end = start + step
        number = math.floor(random.uniform(start, end))
        select_slice_num.append(number)

    select_slice_num.sort()

    slice_content = []
    for slice in select_slice_num:
        select_img_arr = img_arr[slice, :, :]
        re_select_img_arr = transform.resize(select_img_arr, [224, 224], anti_aliasing=True)
        slice_content.append(re_select_img_arr)

    # 转换为3D数组
    slice_content = np.array(slice_content)

    # 将数组转换为 SimpleITK 图像
    new_sitk_img = sitk.GetImageFromArray(slice_content)

    # 保持原始图像的元信息
    new_sitk_img.SetOrigin(sitk_img.GetOrigin())
    new_sitk_img.SetSpacing(sitk_img.GetSpacing())
    new_sitk_img.SetDirection(sitk_img.GetDirection())

    # 保存为 .nii.gz 文件
    (filepath, tempfilename) = os.path.split(ct_path)
    save_img_path_new = tempfilename.replace('.nii.gz', '.nii.gz')
    final_path = os.path.join(save_path, save_img_path_new)

    sitk.WriteImage(new_sitk_img, final_path)
    print(f"Saved: {final_path}")

if __name__ == '__main__':
    K = [40]
    for k_num in K:
        print(f'This is {k_num} instances')
        img_path = r'E:\CT\test\ce'
        save_path = r'E:\CT\MedicalNet-master\MedicalNet5\MedicalNet-master\MedicalNet_pytorch_files\data\MRBrainS18\val'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        seed = 100
        random.seed(seed)
        ct_path = get_dir_ct(img_path)
        ct_path.sort()

        for i in trange(len(ct_path)):
            get_instances(ct_path[i], k_num, save_path)
