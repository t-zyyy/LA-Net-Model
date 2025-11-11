
from lungmask import mask  # lungmask refer:https://github.com/JoHof/lungmask
import SimpleITK as sitk
import os
import tqdm

def get_ct_file(main_path):
    ctpath = []
    # 遍历该文件夹下的所有目录或者文件
    for root, s_dirs, _ in os.walk(main_path, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir) # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)                    # 获取类别文件夹下所有png图片的路径
            for i in range(len(img_list)):
                if img_list[i].endswith('.gz'):
                    path = os.path.join(i_dir, img_list[i])
                    ctpath.append(path)
    return ctpath
def get_listdir(path):#返回具有文件路径的列表
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


if __name__ == '__main__':
    source_path = '/home/zsq/train/pre_process/add_0'
    #ct_path = get_ct_file(source_path)
    ct_path =get_listdir(source_path)
    ct_path.sort()
    #print(len(ct_path))
    #for i in tqdm.trange(len(ct_path)):
    for i in tqdm.trange(163):
        input_path = os.path.join(source_path, ct_path[i])
        input_image = sitk.ReadImage(input_path)
        img_arr = sitk.GetArrayFromImage(input_image)
        segmentation = mask.apply(input_image)
        new_mask_img1 = sitk.GetImageFromArray(segmentation)
        new_mask_img1.SetDirection(input_image.GetDirection())
        new_mask_img1.SetOrigin(input_image.GetOrigin())
        new_mask_img1.SetSpacing(input_image.GetSpacing())
        #new_file_path = ct_path[i].replace('image','lung_mask')
        # 构建新的文件路径
        # 替换 'image' 为 'lung_mask'，并添加适当的扩展名
        new_file_name = ct_path[i].replace('A', 'Amask',1)  # 替换文件夹名或路径中的'image'
        # 构建新的保存路径
        new_file_path = os.path.join('/home/zsq/train/pre_process/mask', new_file_name)
        new_file_path = new_file_path.replace('.gz', '.nii.gz')  # 修改扩展名为 .nii.gz

        # 打印调试信息

        # 确保目标目录存在，如果没有则创建
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

        # 保存新的分割图像
        sitk.WriteImage(new_mask_img1, new_file_path)
        print(f"Image saved to {new_file_path}")
        #print(new_file_name)
