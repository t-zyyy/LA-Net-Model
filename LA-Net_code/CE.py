import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#使系统智能看到编号为1的GPU
#
#
# # #111
# from lungmask import mask
# import SimpleITK as sitk
# import os
# import tqdm
#
# def get_ct_file(main_path):
#     ctpath = []
#     # 遍历该文件夹下的所有目录或者文件
#     for root, s_dirs, _ in os.walk(main_path, topdown=True):  # 获取 train文件下各文件夹名称
#         for sub_dir in s_dirs:
#             i_dir = os.path.join(root, sub_dir) # 获取各类的文件夹 绝对路径
#             img_list = os.listdir(i_dir)                    # 获取类别文件夹下所有png图片的路径
#             for i in range(len(img_list)):
#                 if img_list[i].endswith('.gz'):
#                     path = os.path.join(i_dir, img_list[i])
#                     ctpath.append(path)
#     return ctpath
# def get_listdir(path):#返回具有文件路径的列表
#     tmp_list = []
#     for file in os.listdir(path):
#         if os.path.splitext(file)[1] == '.gz':
#             file_path = os.path.join(path, file)
#             tmp_list.append(file_path)
#     return tmp_list
#
#
# if __name__ == '__main__':
#     source_path = r'/home/zsq/train/pre_process/test'
#     #ct_path = get_ct_file(source_path)
#     ct_path =get_listdir(source_path)
#     ct_path.sort()
#     #print(len(ct_path))
#     #for i in tqdm.trange(len(ct_path)):
#     for i in tqdm.trange(654):
#         input_path = os.path.join(source_path, ct_path[i])
#         input_image = sitk.ReadImage(input_path)
#         input_image = sitk.Cast(input_image, sitk.sitkInt16)  # <== 加上这一行！
#         img_arr = sitk.GetArrayFromImage(input_image)
#         segmentation = mask.apply(input_image)
#         new_mask_img1 = sitk.GetImageFromArray(segmentation)
#         new_mask_img1.SetDirection(input_image.GetDirection())
#         new_mask_img1.SetOrigin(input_image.GetOrigin())
#         new_mask_img1.SetSpacing(input_image.GetSpacing())
#         #new_file_path = ct_path[i].replace('image','lung_mask')
#         # 构建新的文件路径
#         # 替换 'image' 为 'lung_mask'，并添加适当的扩展名
#         new_file_name = ct_path[i].replace('.', 'mask.',1)  # 替换文件夹名或路径中的'image'
#         # 构建新的保存路径
#         new_file_path = os.path.join(r'/home/zsq/train/pre_process/test_mask', new_file_name)
#         new_file_path = new_file_path.replace('.gz', '.nii.gz')  # 修改扩展名为 .nii.gz
#
#         # 打印调试信息
#
#         # 确保目标目录存在，如果没有则创建
#         os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
#
#         # 保存新的分割图像
#         sitk.WriteImage(new_mask_img1, new_file_path)
#         print(f"Image saved to {new_file_path}")
#         #print(new_file_name)
#
#
# from pathlib import Path
#
# # 定义目标路径
# target_dir = Path('/home/zsq/.cache/torch/hub/checkpoints/')
#
# # 如果目录不存在，则创建
# target_dir.mkdir(parents=True, exist_ok=True)
#
# print(f"Directory '{target_dir}' is ready.")
#






# ###########从掩膜中取最大的一块
# #222
# import SimpleITK as sitk
# import os
# import numpy as np
# import tqdm
# import copy
# #import cv2
#
# def get_ct_file(main_path):
#     ctpath = []
#     # 遍历该文件夹下的所有目录或者文件
#     for root, s_dirs, _ in os.walk(main_path, topdown=True):  # 获取 train文件下各文件夹名称
#         for sub_dir in s_dirs:
#             i_dir = os.path.join(root, sub_dir) # 获取各类的文件夹 绝对路径
#             img_list = os.listdir(i_dir)                    # 获取类别文件夹下所有png图片的路径
#             for i in range(len(img_list)):
#                 if img_list[i].endswith('.gz'):
#                     path = os.path.join(i_dir, img_list[i])
#                     ctpath.append(path)
#     return ctpath
# def get_listdir(path):
#     tmp_list = []
#     for file in os.listdir(path):
#         if os.path.splitext(file)[1] == '.gz':
#             file_path = os.path.join(path, file)
#             tmp_list.append(file_path)
#     return tmp_list
# def save_itk(image, origin, spacing, filename):
#     """
#     :param image: images to be saved
#     :param origin: CT origin
#     :param spacing: CT spacing
#     :param filename: save name
#     :return: None
#     """
#     if type(origin) != tuple:
#         if type(origin) == list:
#             origin = tuple(reversed(origin))
#         else:
#             origin = tuple(reversed(origin.tolist()))
#     if type(spacing) != tuple:
#         if type(spacing) == list:
#             spacing = tuple(reversed(spacing))
#         else:
#             spacing = tuple(reversed(spacing.tolist()))
#     itkimage = sitk.GetImageFromArray(image, isVector=False)
#     itkimage.SetSpacing(spacing)
#     itkimage.SetOrigin(origin)
#     sitk.WriteImage(itkimage, filename, True)
#
# def lumTrans_hu(img):
# 	"""
# 	:param img: CT image
# 	:return: Hounsfield Unit window clipped and normalized
# 	"""
# 	lungwin = np.array([-1000.,400.])
# 	newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
# 	newimg[newimg < 0] = 0
# 	newimg[newimg > 1] = 1
# 	newimg = (newimg*255).astype('uint8')
# 	return newimg
#
# def crop_roi(ct,lungmask):
#     lungmask_img = sitk.ReadImage(lungmask)
#     Mask = sitk.GetArrayFromImage(lungmask_img)
#     xx, yy, zz = np.where(Mask)
#
#     box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
#
#     box = np.vstack([np.max([[0, 0, 0], box[:, 0]-5 ], 0), np.min([np.array(Mask.shape), box[:, 1]+5 ], axis=0).T]).T
#     ct_img = sitk.ReadImage(ct)
#     ct = sitk.GetArrayFromImage(ct_img)
#     ct2_img= ct[box[0, 0]:box[0, 1],box[1, 0]:box[1, 1],box[2, 0]:box[2, 1]]
#     return  ct2_img
# def boxx(ct,lungmask):
#     (filepath, tempfilename)=os.path.split(ct)
#     (filename, extension) = os.path.splitext(tempfilename)
#     (filename, extension) = os.path.splitext(filename)
#     lungmask_img = sitk.ReadImage(lungmask)
#     Mask = sitk.GetArrayFromImage(lungmask_img)
#     xx, yy, zz = np.where(Mask)
#
#     box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
#
#     box = np.vstack([np.max([[0, 0, 0], box[:, 0] - 5], 0), np.min([np.array(Mask.shape), box[:, 1] + 5], axis=0).T]).T
#     ct_img = sitk.ReadImage(ct)
#     ct = sitk.GetArrayFromImage(ct_img)
#
#     shapeorg = ct.shape
#     box_shape = np.array([[0, shapeorg[0]], [0, shapeorg[1]], [0, shapeorg[2]]])
#     box = np.concatenate([box, box_shape], axis=0)
#     return box
#
# def savenpy(data_path, lung_mask, prep_folder):
#     """
#     :param data_path: input CT data path
#     :param prep_folder:
#     :return: None
#     """
#     resolution = np.array([1, 1, 1])
#     (filepath, tempfilename) = os.path.split(data_path)
#     (filename, extension) = os.path.splitext(tempfilename)
#     (filename, extension) = os.path.splitext(filename)
#     im = sitk.ReadImage(data_path)
#     img = sitk.GetArrayFromImage(im)
#     mask = sitk.ReadImage(lung_mask)
#     Mask = sitk.GetArrayFromImage(mask)
#     print(img.shape, end=" ")
#     # im, m1, m2, mtotal, origin, spacing = step1_python(data_path)
#     # print('Origin: ', origin, ' Spacing: ', spacing, 'img shape: ', im.shape)
#     # Mask = m1 + m2
#     xx, yy, zz = np.where(Mask)
#     box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
#     margin = 0
#
#     box = np.vstack(
#         [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([np.array(Mask.shape), box[:, 1] + margin], axis=0).T]).T
#
#     # save the lung mask
#     img[np.isnan(img)] = -2000
#     sliceim_hu = lumTrans_hu(Mask)
#     shapeorg = sliceim_hu.shape
#     box_shape = np.array([[0, shapeorg[0]], [0, shapeorg[1]], [0, shapeorg[2]]])
#
#
#     Mask_crop = Mask[box_shape[0, 0]:box_shape[0, 1],
#                 box[1, 0]:box[1, 1],
#                 box[2, 0]:box[2, 1]]
#
#     # save_itk(Mask_crop ,im.GetOrigin(), im.GetSpacing(), os.path.join(prep_folder, filename + "_lungmask_hu.nii.gz"))
#     # convex_mask = m1
#     # dm1 = process_mask(m1)
#     # dm2 = process_mask(m2)
#     # dilatedMask = (dm1 + dm2) | mtotal
#     # Mask = m1+m2
#     # dilatedMask = dilatedMask.astype('uint8')
#     # Mask = Mask.astype('uint8')
#
#     # save the CT image
#     sliceim2_hu = img[box[0, 0]:box[0, 1],
#                   box[1, 0]:box[1, 1],
#                   box[2, 0]:box[2, 1]]
#     box[0, 1] = box_shape[0, 1]
#     box[0, 0] = box_shape[0, 0]
#     # save box (image original shape and cropped window region)
#     box = np.concatenate([box, box_shape], axis=0)
#     # np.save(os.path.join(prep_folder, filename + '_box.npy'), box)
#     # save processed image
#     print(" -> ", sliceim2_hu.shape, end=" ")
#
#     save_itk(sliceim2_hu, im.GetOrigin(), im.GetSpacing(),os.path.join(prep_folder, filename + ".nii.gz"))
#     return
#
#
# if __name__ == '__main__':
#     # crop image path
#
#     #img_path = '/disk2/wuyanan/wanren/identification/SSL/data'
#     img_path =r'/home/zsq/train/pre_process/train'#
#     # crop image path`
#     # label_path ='/disk2/wuyanan/wanren/identification/training/data/0'
#     # the lobe or lung mask path for cropping image and label
#
#     #lung_mask_path = '/disk2/wuyanan/wanren/identification/SSL/lobe'
#     lung_mask_path =r'/home/zsq/train/pre_process/train_mask'#
#     # save path
#     #save_path = '/disk1/wuyanan/wanren/identification/lung_box/ssl'
#     save_path =r'/home/zsq/train/pre_process/train_save'#
#
#     img_path = get_listdir(img_path)
#     img_path.sort()
#     # label_path = get_listdir(label_path)
#     # label_path.sort()
#     l_mask = get_listdir(lung_mask_path)
#     l_mask.sort()
#
#     for i in tqdm.trange(len(img_path)):#
#         (filepath, tempfilename) = os.path.split(img_path[i])
#         (filename, extension) = os.path.splitext(tempfilename)
#         (filename, extension) = os.path.splitext(filename)
#         savenpy(img_path[i], l_mask[i], save_path)











##############从掩膜中只取肺区
# # #222
# import SimpleITK as sitk
# import os
# import numpy as np
# import tqdm
# import copy
# #import cv2
#
# def get_ct_file(main_path):
#     ctpath = []
#     # 遍历该文件夹下的所有目录或者文件
#     for root, s_dirs, _ in os.walk(main_path, topdown=True):  # 获取 train文件下各文件夹名称
#         for sub_dir in s_dirs:
#             i_dir = os.path.join(root, sub_dir) # 获取各类的文件夹 绝对路径
#             img_list = os.listdir(i_dir)                    # 获取类别文件夹下所有png图片的路径
#             for i in range(len(img_list)):
#                 if img_list[i].endswith('.gz'):
#                     path = os.path.join(i_dir, img_list[i])
#                     ctpath.append(path)
#     return ctpath
# def get_listdir(path):
#     tmp_list = []
#     for file in os.listdir(path):
#         if os.path.splitext(file)[1] == '.gz':
#             file_path = os.path.join(path, file)
#             tmp_list.append(file_path)
#     return tmp_list
# def save_itk(image, origin, spacing, filename):
#     """
#     :param image: images to be saved
#     :param origin: CT origin
#     :param spacing: CT spacing
#     :param filename: save name
#     :return: None
#     """
#     if type(origin) != tuple:
#         if type(origin) == list:
#             origin = tuple(reversed(origin))
#         else:
#             origin = tuple(reversed(origin.tolist()))
#     if type(spacing) != tuple:
#         if type(spacing) == list:
#             spacing = tuple(reversed(spacing))
#         else:
#             spacing = tuple(reversed(spacing.tolist()))
#     itkimage = sitk.GetImageFromArray(image, isVector=False)
#     itkimage.SetSpacing(spacing)
#     itkimage.SetOrigin(origin)
#     sitk.WriteImage(itkimage, filename, True)
#
# def lumTrans_hu(img):
# 	"""
# 	:param img: CT image
# 	:return: Hounsfield Unit window clipped and normalized
# 	"""
# 	lungwin = np.array([-1000.,400.])
# 	newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
# 	newimg[newimg < 0] = 0
# 	newimg[newimg > 1] = 1
# 	newimg = (newimg*255).astype('uint8')
# 	return newimg
#
# def crop_roi(ct,lungmask):
#     lungmask_img = sitk.ReadImage(lungmask)
#     Mask = sitk.GetArrayFromImage(lungmask_img)
#     xx, yy, zz = np.where(Mask)
#
#     box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
#
#     box = np.vstack([np.max([[0, 0, 0], box[:, 0]-5 ], 0), np.min([np.array(Mask.shape), box[:, 1]+5 ], axis=0).T]).T
#     ct_img = sitk.ReadImage(ct)
#     ct = sitk.GetArrayFromImage(ct_img)
#     ct2_img= ct[box[0, 0]:box[0, 1],box[1, 0]:box[1, 1],box[2, 0]:box[2, 1]]
#     return  ct2_img
# def boxx(ct,lungmask):
#     (filepath, tempfilename)=os.path.split(ct)
#     (filename, extension) = os.path.splitext(tempfilename)
#     (filename, extension) = os.path.splitext(filename)
#     lungmask_img = sitk.ReadImage(lungmask)
#     Mask = sitk.GetArrayFromImage(lungmask_img)
#     xx, yy, zz = np.where(Mask)
#
#     box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
#
#     box = np.vstack([np.max([[0, 0, 0], box[:, 0] - 5], 0), np.min([np.array(Mask.shape), box[:, 1] + 5], axis=0).T]).T
#     ct_img = sitk.ReadImage(ct)
#     ct = sitk.GetArrayFromImage(ct_img)
#
#     shapeorg = ct.shape
#     box_shape = np.array([[0, shapeorg[0]], [0, shapeorg[1]], [0, shapeorg[2]]])
#     box = np.concatenate([box, box_shape], axis=0)
#     return box
#
#
# def savenpy(data_path, lung_mask, prep_folder):
#     """
#     :param data_path: input CT data path
#     :param lung_mask: binary lung mask path
#     :param prep_folder: output folder to save masked CT image
#     :return: None
#     """
#     (filepath, tempfilename) = os.path.split(data_path)
#     (filename, extension) = os.path.splitext(tempfilename)
#     (filename, extension) = os.path.splitext(filename)
#
#     # 读取原始CT图像和mask
#     im = sitk.ReadImage(data_path)
#     img = sitk.GetArrayFromImage(im)
#     mask = sitk.ReadImage(lung_mask)
#     Mask = sitk.GetArrayFromImage(mask)
#
#     # 防止NaN
#     img[np.isnan(img)] = -2000
#
#     # 二值化mask（保险起见）
#     Mask = (Mask > 0).astype(np.uint8)
#
#     print(img.shape, end=" ")
#
#     # 定位mask的bounding box
#     xx, yy, zz = np.where(Mask)
#     box = np.array([[np.min(xx), np.max(xx)],
#                     [np.min(yy), np.max(yy)],
#                     [np.min(zz), np.max(zz)]])
#
#     margin = 0  # 可选边缘扩展
#     box = np.vstack([
#         np.max([[0, 0, 0], box[:, 0] - margin], 0),
#         np.min([np.array(Mask.shape), box[:, 1] + margin], axis=0).T
#     ]).T
#
#     # 裁剪图像和掩膜
#     crop_img = img[box[0, 0]:box[0, 1],
#                box[1, 0]:box[1, 1],
#                box[2, 0]:box[2, 1]]
#
#     crop_mask = Mask[box[0, 0]:box[0, 1],
#                 box[1, 0]:box[1, 1],
#                 box[2, 0]:box[2, 1]]
#
#     # 只保留肺部区域像素
#     masked_ct = np.where(crop_mask == 1, crop_img, -2000)
#
#     print(" -> ", masked_ct.shape, end=" ")
#
#     # 保存
#     save_itk(masked_ct, im.GetOrigin(), im.GetSpacing(), os.path.join(prep_folder, filename + ".nii.gz"))
#
#     return
#
#
# if __name__ == '__main__':
#     # crop image path
#
#     #img_path = '/disk2/wuyanan/wanren/identification/SSL/data'
#     img_path =r'/home/zsq/train/pre_process/test'#
#     # crop image path`
#     # label_path ='/disk2/wuyanan/wanren/identification/training/data/0'
#     # the lobe or lung mask path for cropping image and label
#
#     #lung_mask_path = '/disk2/wuyanan/wanren/identification/SSL/lobe'
#     lung_mask_path =r'/home/zsq/train/pre_process/test_mask'#
#     # save path
#     #save_path = '/disk1/wuyanan/wanren/identification/lung_box/ssl'
#     save_path =r'/home/zsq/train/pre_process/test_纯肺'#
#
#     img_path = get_listdir(img_path)
#     img_path.sort()
#     # label_path = get_listdir(label_path)
#     # label_path.sort()
#     l_mask = get_listdir(lung_mask_path)
#     l_mask.sort()
#
#     for i in tqdm.trange(len(img_path)):#
#         (filepath, tempfilename) = os.path.split(img_path[i])
#         (filename, extension) = os.path.splitext(tempfilename)
#         (filename, extension) = os.path.splitext(filename)
#         savenpy(img_path[i], l_mask[i], save_path)












############提取层数

# #
# # ##333
import os
import numpy as np
import SimpleITK as sitk
from tqdm import trange
import random
import math
import time
#import cv2
from random import randint
from skimage import transform
import numpy as np



from PIL import Image,ImageOps
# import matplotlib.pyplot as plt



def get_dir_ct(main_path):
    ctpath = []
    ct_list = os.listdir(main_path)  # 列出文件夹下所有的目录与文件
    # 遍历该文件夹下的所有目录或者文件
    for ii in range(0, len(ct_list)):
        path = os.path.join(main_path, ct_list[ii])
        ctpath.append(path)
    return ctpath

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

def get_instances(ct_path,K,save_path):
    sitk_img = sitk.ReadImage(ct_path)
    img_arr = sitk.GetArrayFromImage(sitk_img)
    slice_num = img_arr.shape[0]
    # select_slice_num = []

    # 设定随机数生成的范围
    min_val = 0
    max_val = slice_num

    # 计算去掉5%区间的新范围
    cut_off = 0.05 * (max_val - min_val)#为了避免选择图像的边缘区域，选择时会将有效范围向内裁剪 5%。
    new_min = min_val + cut_off
    new_max = max_val - cut_off

    # 将新区间平均分成20份
    step = (new_max - new_min) / K#选取K个切片保存

    # 生成20个随机数
    select_slice_num = []
    for i in range(K):
        # 计算每一份的小区间的起始和结束值
        start = new_min + i * step
        end = start + step
        # 从每个小区间中生成一个随机数
        number = math.floor(random.uniform(start, end))
        select_slice_num.append(number)

    # # 设置随机数生成的范围
    # lower_bound = int(slice_num * 0.05)
    # upper_bound = int(slice_num * 0.95)
    #
    # # 生成k个随机数
    # select_slice_num = random.sample(range(lower_bound, upper_bound + 1), K)
    select_slice_num.sort()
    # for i in range(K):
    #     select_slice_num.append(randint(math.floor(i/K*slice_num),math.floor((i+1)/K*slice_num)-1))
    slice_content = []
    for slice in select_slice_num:
        select_img_arr = img_arr[slice,:,:]

        # img = Image.fromarray(select_img_arr.astype('int16'))
        # ivt_image = ImageOps.invert(img)
        # padding = (0, 0, 0, 0)
        # # 如果担心检测出来的bbox过小，可以加点padding
        # bbox = ivt_image.getbbox()
        # left = bbox[0] - padding[0]
        # top = bbox[1] - padding[1]
        # right = bbox[2] + padding[2]
        # bottom = bbox[3] + padding[3]
        # cropped_image = img.crop([left, top, right, bottom])
        # cropped_image = np.array(cropped_image)

        re_select_img_arr = transform.resize(select_img_arr, [224, 224])

        # min_bound = -1000
        # max_bound = 500
        # re_select_img_arr[re_select_img_arr<min_bound] = min_bound
        # re_select_img_arr[re_select_img_arr>max_bound] = max_bound
        # re_select_img_arr = (re_select_img_arr-min_bound)*255.0/(max_bound-min_bound)
        # 保存每一张slice为一个文件
        # slice_content = np.array(re_select_img_arr)
        # (filepath, tempfilename) = os.path.split(ct_path)
        # save_img_path_new = tempfilename.replace('.nii.gz', '_'+str(slice)+'.npy')
        # final_path = os.path.join(save_path, save_img_path_new)
        #
        # np.save(final_path, slice_content)

        slice_content.append(re_select_img_arr)
    slice_content = np.array(slice_content)

    print(slice_content.shape)
    # save_img_path = ct_path.replace('data','instance')
    # save_img_path_new = save_img_path.replace('nii.gz','npy')
    (filepath, tempfilename) = os.path.split(ct_path)
    save_img_path_new = tempfilename.replace('nii.gz','npy')
    final_path = os.path.join(save_path, save_img_path_new)

    np.save(final_path,slice_content)



if __name__ == '__main__':
    # aaa = np.load('1.npy')
    K = [20]
    for k_num in K:
        print(f'This is {k_num} instances')
        img_path = r'/disk2/zsq/train/pre_process/test'
        save_path = r'/disk2/zsq/train/pre_process/test20'
        # save_path = os.path.join('/disk1/wuyanan/wanren/identification/SSL',str(k_num))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save_path = r'G:\3-airway-and-lungfield-paper\9-instances\COPD\dalian\20'
        # seed = 100  #####之前是100
        seed = int(time.time()) % 10000  # 根据当前时间生成一个随机种子

        random.seed(seed)
        ct_path = get_dir_ct(img_path)
        ct_path.sort()
        for i in trange(len(ct_path)):
            get_instances(ct_path[i],k_num,save_path)


















import os

# def add_prefix_to_files(folder_path, prefix='A'):
#     for filename in os.listdir(folder_path):
#         old_path = os.path.join(folder_path, filename)
#
#         # 跳过文件夹，只处理文件
#         if not os.path.isfile(old_path):
#             continue
#
#         # 如果已经有前缀了，就跳过
#         if filename.startswith(prefix):
#             continue
#
#         new_filename = prefix + filename
#         new_path = os.path.join(folder_path, new_filename)
#
#         os.rename(old_path, new_path)
#         print(f"Renamed: {filename} → {new_filename}")
#
#
# # 使用示例
# folder = "/home/zsq/train/external_shengjing/buchong测试"  # 替换为你的文件夹路径
# add_prefix_to_files(folder)





# ############333333  取完全不同的另60切片
# import os
# import numpy as np
# import SimpleITK as sitk
# from tqdm import trange
# import random
# import math
# from skimage import transform
#
#
# def get_dir_ct(main_path):
#     ctpath = []
#     ct_list = os.listdir(main_path)  # 列出文件夹下所有的目录与文件
#     for ii in range(0, len(ct_list)):
#         path = os.path.join(main_path, ct_list[ii])
#         ctpath.append(path)
#     return ctpath
#
#
# def get_instances(ct_path, K, save_path_primary, save_path_secondary):
#     sitk_img = sitk.ReadImage(ct_path)
#     img_arr = sitk.GetArrayFromImage(sitk_img)
#     slice_num = img_arr.shape[0]
#
#     # 设定范围
#     min_val = 0
#     max_val = slice_num
#
#     # 避免边缘区域
#     cut_off = 0.05 * (max_val - min_val)
#     new_min = int(min_val + cut_off)
#     new_max = int(max_val - cut_off)
#
#     # 将所有切片分成 K 份，计算每一份的大小
#     step = (new_max - new_min) / K
#     slice_groups = []
#
#     # 分配切片，分成 K 份
#     for i in range(K):
#         start = int(new_min + i * step)
#         end = min(int(start + step), slice_num)  # 确保 end 不会超出切片总数
#         slice_groups.append(list(range(start, end)))
#
#     # 生成主集
#     primary_slices = []
#     group_to_primary_map = {}  # 记录每个 group 对应的 primary_slice
#
#     for group in slice_groups:
#         if len(group) == 1:
#             primary_slices.append(group[0])
#             group_to_primary_map[tuple(group)] = group[0]
#         elif len(group) > 1:
#             number = random.choice(group)
#             primary_slices.append(number)
#             group_to_primary_map[tuple(group)] = number
#
#     primary_slices.sort()
#
#     # 生成次集
#     secondary_slices = []
#     for group in slice_groups:
#         remaining_slices = [slice for slice in group if slice not in primary_slices]
#         if remaining_slices:
#             number = random.choice(remaining_slices)
#             secondary_slices.append(number)
#         else:
#             # 如果没有剩余切片，从主集中选择对应的切片
#             secondary_slices.append(group_to_primary_map.get(tuple(group), None))
#
#     secondary_slices = [x for x in secondary_slices if x is not None]  # 移除任何可能的 None 值
#     secondary_slices.sort()
#
#     # 保存主集切片
#     save_slices(img_arr, primary_slices, save_path_primary, ct_path)
#
#     # 保存次集切片
#     save_slices(img_arr, secondary_slices, save_path_secondary, ct_path)
#
#
# def save_slices(img_arr, slice_indices, save_path, ct_path):
#     slice_content = []
#     for slice_idx in slice_indices:
#         select_img_arr = img_arr[slice_idx, :, :]
#         resized_img = transform.resize(select_img_arr, [224, 224], preserve_range=True)
#         slice_content.append(resized_img)
#
#     slice_content = np.array(slice_content)
#
#     # 保存切片内容
#     _, tempfilename = os.path.split(ct_path)
#     save_img_path_new = tempfilename.replace('nii.gz', 'npy')
#     final_path = os.path.join(save_path, save_img_path_new)
#     np.save(final_path, slice_content)
#
#
# if __name__ == '__main__':
#     K = 60
#     img_path = r'/home/zsq/train/pre_process/test'
#     save_path_primary = r'/home/zsq/train/pre_process/test60_1'
#     save_path_secondary = r'/home/zsq/train/pre_process/test60_2'
#
#     # 创建保存路径
#     for path in [save_path_primary, save_path_secondary]:
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#     # 固定随机种子
#     seed = 100
#     random.seed(seed)
#
#     # 获取所有 CT 文件路径
#     ct_path_list = get_dir_ct(img_path)
#     ct_path_list.sort()
#
#     # 处理每个 CT 文件
#     for i in trange(len(ct_path_list)):
#         try:
#             get_instances(ct_path_list[i], K, save_path_primary, save_path_secondary)
#         except Exception as e:
#             print(f"Error processing file {ct_path_list[i]}: {e}")













# ###############根据Excel文件中的文件名列表，从源文件夹中查找并复制指定文件到目标文件夹。
#
# import os
# import shutil
# import pandas as pd
#
#
# def copy_files_based_on_excel(excel_path, source_dir, target_dir, column_name='A'):
#     """
#     根据Excel文件中的文件名列表，从source_dir中查找并复制对应的文件到target_dir。
#
#     :param excel_path: Excel 文件路径
#     :param source_dir: 源文件夹路径，包含所有文件
#     :param target_dir: 目标文件夹路径，存放筛选出的文件
#     :param column_name: Excel 中文件名列的名称，默认为 'A'
#     """
#     # 读取Excel文件
#     df = pd.read_excel(excel_path, usecols=[column_name])
#
#     # 确保目标文件夹存在
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
#     # 遍历Excel中的文件名，并尝试复制文件
#     for index, row in df.iterrows():
#         file_name = str(row[column_name]).strip()
#         source_file = os.path.join(source_dir, file_name)
#
#         if os.path.isfile(source_file):
#             shutil.copy2(source_file, target_dir)  # 使用copy2以保留元数据
#             print(f"Copied {file_name}")
#         else:
#             print(f"File not found: {file_name}")
#
#
# if __name__ == "__main__":
#     # 设置参数
#     excel_path = r'/home/zsq/train/pre_process/PRI.xlsx'  # 替换为你的Excel文件路径
#     source_dir = r'/home/zsq/train/pre_process/5378'  # 替换为源文件夹路径
#     target_dir = r'/home/zsq/train/pre_process/PRI'  # 替换为目标文件夹路径
#
#     # 执行复制操作
#     copy_files_based_on_excel(excel_path, source_dir, target_dir)
#
#








#
# #############一个excel中有的文件名，不能不现在另一个excel中（要在另一个excel中删除
# import pandas as pd
#
# # 定义文件路径
# file_a_path = r"E:/CT/挑选/PRI.xlsx" # 存放文件名的Excel文件
# file_b_path = r"E:/CT/挑选/CT.xlsx"  # 需要修改的Excel文件
# output_path = r"E:/CT/挑选/xin.xlsx"  # 保存修改后的Excel文件
#
# # 读取Excel文件
# df_a = pd.read_excel(file_a_path, usecols=[0])  # 只读取第一列的文件名
# df_b = pd.read_excel(file_b_path)
#
# # 获取文件名列表
# file_names_a = df_a.iloc[:, 0].tolist()  # 从第一列获取文件名列表
#
# # 从df_b中删除在df_a中出现的文件名及对应的第二列
# df_b_filtered = df_b[~df_b.iloc[:, 0].isin(file_names_a)]
#
# # 保存结果到新的Excel文件
# df_b_filtered.to_excel(output_path, index=False)
#
# print(f"已生成修改后的文件: {output_path}")
#
#




# ##############按照8：2分配数据（excel)
#
# #####################excle按照8：2分割
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# def split_excel(input_file, output_file_1, output_file_2, ratio=0.8):
#     """
#     按指定比例随机分割Excel文件的行并保存到两个文件中。
#
#     :param input_file: 输入的Excel文件路径
#     :param output_file_1: 保存第一部分的Excel文件路径
#     :param output_file_2: 保存第二部分的Excel文件路径
#     :param ratio: 第一部分的比例（默认 0.7 表示 7:3）
#     """
#     # 读取 Excel 文件
#     df = pd.read_excel(input_file, engine='openpyxl')
#
#     # 使用 train_test_split 随机划分数据
#     df_part1, df_part2 = train_test_split(df, test_size=(1 - ratio), random_state=42)
#
#     # 保存到两个新的 Excel 文件
#     df_part1.to_excel(output_file_1, index=False)
#     df_part2.to_excel(output_file_2, index=False)
#
#     print(f"数据已按 {ratio*100:.0f}:{(1-ratio)*100:.0f} 的比例分割完成。")
#     print(f"第一部分保存到: {output_file_1}")
#     print(f"第二部分保存到: {output_file_2}")
#
# # 示例使用
# input_file_path = r"/home/zsq/train/pre_process/zong.xlsx"  # 输入文件路径
# output_file_1_path = r"/home/zsq/train/pre_process/train.xlsx"  # 第一部分保存路径
# output_file_2_path = r"/home/zsq/train/pre_process/test.xlsx"  # 第二部分保存路径
#
# split_excel(input_file_path, output_file_1_path, output_file_2_path, ratio=0.8)













###########用于根据 Excel 文件中 A 列的文件名，将某文件夹中符合条件的文件复制到另一个文件夹
import os
import shutil
import pandas as pd


# def copy_files_from_excel(excel_path, source_folder, target_folder):
#     """
#     根据 Excel 文件中 A 列的文件名，从 source_folder 中复制文件到 target_folder。
#
#     :param excel_path: Excel 文件路径
#     :param source_folder: 源文件夹路径
#     :param target_folder: 目标文件夹路径
#     """
#     # 确保目标文件夹存在
#     os.makedirs(target_folder, exist_ok=True)
#
#     # 读取 Excel 文件中的 A 列
#     df = pd.read_excel(excel_path, usecols=[0])  # 假设文件名在 A 列
#     file_names = df.iloc[:, 0].dropna().astype(str).tolist()
#
#     # 遍历文件名列表
#     for file_name in file_names:
#         source_file = os.path.join(source_folder, file_name)
#         if os.path.isfile(source_file):
#             shutil.copy(source_file, target_folder)
#             print(f"已复制: {file_name}")
#         else:
#             print(f"文件未找到: {file_name}")
#
#
# if __name__ == "__main__":
#     # 修改为你的实际路径
#     excel_path = r"/home/zsq/train/pre_process/train.xlsx"  # Excel 文件路径
#     source_folder = r"/home/zsq/train/pre_process/5378"
#     target_folder = r"/home/zsq/train/pre_process/train"
#
#     copy_files_from_excel(excel_path, source_folder, target_folder)
#
#
# import pandas as pd








####从CT-PFT文件夹对应位置对应数据
# #
# import pandas as pd
#
# def update_excel1_from_excel2(excel1_path, excel2_path, output_path):
#     # 读取两个Excel文件
#     df1 = pd.read_excel(excel1_path)
#     df2 = pd.read_excel(excel2_path)
#
#     # 将Excel2的A列作为索引，映射C列数据
#     mapping = df2.set_index('J')['身高/m'].to_dict()
#
#     # 将Excel1的A列文件名后缀从.npy修改为.nii.gz
#     df1['A'] = df1['A'].apply(lambda x: x.replace('.npy', '.nii.gz') if isinstance(x, str) else x)
#
#     # 根据Excel1的A列匹配Excel2的C列数据
#     df1['M'] = df1['A'].map(mapping)
#
#     # 保存更新后的Excel1
#     df1.to_excel(output_path, index=False)
#
#     print(f'更新完成，结果已保存至 {output_path}')
#
#
# # 示例调用
# update_excel1_from_excel2('/home/zsq/train/pre_process/DATE/外部验证excel/其余关系/COPD校正偏差/抽取.xlsx',
#                           '/home/zsq/train/pre_process/DATE/外部验证excel/盛京/2次肺功能.xlsx',#/home/zsq/train/pre_process/CT-PFT.xlsx
#                           '/home/zsq/train/pre_process/DATE/外部验证excel/其余关系/COPD校正偏差/抽取.xlsx')
#







# import pandas as pd
#
# def sample_from_excel(input_file, output_file, n):
#     try:
#         # 读取 Excel 文件
#         df = pd.read_excel(input_file)
#
#         # 检查样本量是否超过数据行数
#         if n > len(df):
#             print(f"样本量 {n} 超过了数据总行数 {len(df)}，将全部导出。")
#             n = len(df)
#
#         # 随机抽取 n 个样本（不放回）
#         sample = df.sample(n=n, random_state=None)  # random_state=None 表示每次都随机
#
#         # 将样本保存到新的 Excel 文件中
#         sample.to_excel(output_file, index=False)
#
#         print(f"已成功抽取 {n} 个样本，保存到 {output_file}")
#
#     except Exception as e:
#         print(f"发生错误：{e}")
#
# if __name__ == "__main__":
#     input_file = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/COPD/COPD严重程度/G2.xlsx"  # 替换成你的输入文件路径
#     output_file = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/COPD/COPD严重程度/G2抽.xlsx"  # 输出文件名
#     n = 41  # 需要抽取的样本数量
#
#     sample_from_excel(input_file, output_file, n)







# 测试每个网络在每个年龄段的平局误差
# #
# import numpy as np
# from torch.utils.data import DataLoader
# from setting import parse_opts
# from brains import BrainS18Dataset
# from model import generate_model
# import pandas as pd
# import os
# import torch
# from tqdm import tqdm
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# np.seterr(all='ignore')
#
#
# def run_ce(data_loader, model, img_names, labels):
#     predictions = []
#     model.eval()
#
#     with torch.no_grad():
#         for batch_id, batch_data in enumerate(tqdm(data_loader, desc="Processing Samples", unit="batch")):
#             volume, _ = batch_data
#             volume = volume.cuda() if torch.cuda.is_available() else volume.cpu()
#             pred = model(volume).cpu().numpy().flatten()
#             predictions.append(pred[0])
#             tqdm.write(f"Batch {batch_id + 1}: Predicted Age = {pred[0]:.2f}, True Age = {labels[batch_id]:.2f}")
#
#     deviations = [pred - true for pred, true in zip(predictions, labels)]
#     return predictions, deviations
#
#
# def save_results_to_excel(results, output_path):
#     df = pd.DataFrame(results, columns=["Model", "Mean Deviation"])
#     df.to_excel(output_path, index=False)
#     print(f"Results saved to {output_path}")
#
#
# def main():
#     sets = parse_opts()
#     sets.target_type = "age_regression"
#     sets.phase = 'test'
#     sets.test_root = '/home/zsq/train/COPD_60'
#     # sets.test_root = '/home/zsq/train/pre_process/test60'
#     # sets.test_file = '/home/zsq/train/pre_process/DATE/linshi3/age_range_50_55.xlsx'
#     sets.test_file = '/home/zsq/train/pre_process/DATE/COPD分段/age_range_50_55.xlsx'
#     weight_folder = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_9_4(50_55)/resnet_34'
#     output_excel = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_9_4(50_55)/resnet_34/COPD50_55.xlsx'
#     n = 11  # 指定测试前 n 个网络
#
#     testing_data = BrainS18Dataset(sets.test_root, sets.test_file, sets)
#     df = pd.read_excel(sets.test_file)
#     img_names = df.iloc[:, 0].tolist()
#     labels = df.iloc[:, 1].tolist()
#
#     results = []
#     weight_files = sorted([f for f in os.listdir(weight_folder) if f.endswith(".pth.tar")])[:n]
#
#     for weight_file in weight_files:
#         weight_path = os.path.join(weight_folder, weight_file)
#         print(f"Processing model: {weight_path}")
#
#         checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
#         model_state_dict = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
#         model, _ = generate_model(sets)
#         model.load_state_dict(model_state_dict)
#         model = model.cuda() if torch.cuda.is_available() else model.cpu()
#
#         data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
#         predictions, deviations = run_ce(data_loader, model, img_names, labels)
#         mean_deviation = np.mean(deviations)
#
#         print(f"Model {weight_file}: Mean Deviation = {mean_deviation:.4f}")
#         for img, pred, dev in zip(img_names, predictions, deviations):
#             print(f"{img}: Predicted = {pred:.2f}, Deviation = {dev:.2f}")
#
#         results.append([weight_file, mean_deviation])
#
#     save_results_to_excel(results, output_excel)
#
#
# if __name__ == "__main__":
#     main()
#










#
#
# ########同时验证一个网络在两个验证集的效果
# import numpy as np
# from torch.utils.data import DataLoader
# from setting import parse_opts
# from brains import BrainS18Dataset
# from model import generate_model
# import pandas as pd
# import os
# import torch
# from tqdm import tqdm
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# np.seterr(all='ignore')
#
# def run_ce_MAE(data_loader, model, labels):
#     predictions = []
#     model.eval()
#
#     with torch.no_grad():
#         for batch_id, batch_data in enumerate(tqdm(data_loader, desc="Processing Samples", unit="batch")):
#             volume, _ = batch_data
#             volume = volume.cuda() if torch.cuda.is_available() else volume.cpu()
#             pred = model(volume).cpu().numpy().flatten()
#             predictions.append(pred[0])
#
#     # deviations = [pred - true for pred, true in zip(predictions, labels)]
#     deviations = [abs(pred - true) for pred, true in zip(predictions, labels)]
#     return np.mean(deviations)
#
# def run_ce(data_loader, model, labels):
#     predictions = []
#     model.eval()
#
#     with torch.no_grad():
#         for batch_id, batch_data in enumerate(tqdm(data_loader, desc="Processing Samples", unit="batch")):
#             volume, _ = batch_data
#             volume = volume.cuda() if torch.cuda.is_available() else volume.cpu()
#             pred = model(volume).cpu().numpy().flatten()
#             predictions.append(pred[0])
#
#     deviations = [pred - true for pred, true in zip(predictions, labels)]
#     # deviations = [abs(pred - true) for pred, true in zip(predictions, labels)]
#     return np.mean(deviations)
#
#
# def save_results_to_excel(results, output_path):
#     df = pd.DataFrame(results, columns=["Model", "Mean Deviation Set 1", "Mean Deviation Set 2","MAE1","MAE2"])
#     df.to_excel(output_path, index=False)
#     print(f"Results saved to {output_path}")
#
#
# def main():
#     sets = parse_opts()
#     sets.target_type = "age_regression"
#     sets.phase = 'test'
#
#     sets.test_root_1 = '/home/zsq/train/COPD_60'
#     # sets.test_file_1 = '/home/zsq/train/pre_process/DATE/符合分布的train和test/COPD.xlsx'
#     sets.test_file_1 = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/COPD/女/age_range_1_47.xlsx'
#
#     sets.test_root_2 = '/home/zsq/train/pre_process/test60'
#     # sets.test_file_2 = '/home/zsq/train/pre_process/DATE/符合分布的train和test/test.xlsx'
#     sets.test_file_2 = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/test/test性.xlsx'
#
#     weight_folder = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_10_1/protect'
#     output_excel = '/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_10_1/protect/分析.xlsx'
#     n = 1  # 指定测试前 n 个网络
#
#     # 读取数据集1
#     testing_data_1 = BrainS18Dataset(sets.test_root_1, sets.test_file_1, sets)
#     df1 = pd.read_excel(sets.test_file_1)
#     labels_1 = df1.iloc[:, 1].tolist()
#     data_loader_1 = DataLoader(testing_data_1, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
#
#     # 读取数据集2
#     testing_data_2 = BrainS18Dataset(sets.test_root_2, sets.test_file_2, sets)
#     df2 = pd.read_excel(sets.test_file_2)
#     labels_2 = df2.iloc[:, 1].tolist()
#     data_loader_2 = DataLoader(testing_data_2, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
#
#     results = []
#     weight_files = sorted([f for f in os.listdir(weight_folder) if f.endswith(".pth.tar")])[:n]
#
#     for weight_file in weight_files:
#         weight_path = os.path.join(weight_folder, weight_file)
#         print(f"Processing model: {weight_path}")
#
#         checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
#         model_state_dict = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
#         model, _ = generate_model(sets)
#         model.load_state_dict(model_state_dict)
#         model = model.cuda() if torch.cuda.is_available() else model.cpu()
#
#         mean_deviation_1 = run_ce(data_loader_1, model, labels_1)
#         mean_deviation_2 = run_ce(data_loader_2, model, labels_2)
#         mean_deviation_1_MAE = run_ce_MAE(data_loader_1, model, labels_1)
#         mean_deviation_2_MAE = run_ce_MAE(data_loader_2, model, labels_2)
#         print(
#             f"Model {weight_file}: Mean Deviation Set 1 = {mean_deviation_1:.4f}, Mean Deviation Set 2 = {mean_deviation_2:.4f},MAE Set 1 = {mean_deviation_1_MAE:.4f}, MAE Set 2 = {mean_deviation_2_MAE:.4f}"
#         )
#
#         results.append([weight_file, mean_deviation_1, mean_deviation_2, mean_deviation_1_MAE, mean_deviation_2_MAE])
#
#     save_results_to_excel(results, output_excel)
#
#
# if __name__ == "__main__":
#     main()
#











#
# ####从excel选取，保证年龄分布一致
# import pandas as pd
# import random
# import os
#
# # 读取 Excel 文件
# input_file = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/COPD严重程度/test偏差.xlsx'  # 输入文件路径
# df = pd.read_excel(input_file)
#
# # 确保数据按列加载正确
# df.columns = ['filename', 'age']
#
# # 设定年龄区间和比例
# age_groups = {
#     '35-39': (42, 49),
#     '40-44': (50, 56),
#     '45-49': (57, 62),
#     '50-54': (63, 74)
# }
#
# proportions = {
#     '35-39': 9/40,
#     '40-44': 6/40,
#     '45-49': 14/40,
#     '50-54': 11/40
#
# }
#
# # 按年龄分组
# grouped = {group: df[(df['age'] >= age_groups[group][0]) & (df['age'] <= age_groups[group][1])]
#            for group in age_groups}
#
# # 计算每个年龄段要抽取的数量
# total_rows = len(df)
# extract_counts = {group: int(total_rows * proportions[group]) for group in age_groups}
#
# # 存储抽取后的数据
# df_A = pd.DataFrame(columns=['filename', 'age'])
# df_B = pd.DataFrame(columns=['filename', 'age'])
#
# # 按比例抽取数据
# for group in age_groups:
#     group_data = grouped[group]
#     extract_count = extract_counts[group]
#
#     # 如果该组数据数量大于要抽取的数量，则随机抽取
#     if len(group_data) >= extract_count:
#         extracted = group_data.sample(n=extract_count, random_state=42)
#         df_A = pd.concat([df_A, extracted], ignore_index=True)
#         remaining = group_data.drop(extracted.index)
#     else:
#         # 如果不足抽取数量，取所有数据
#         df_A = pd.concat([df_A, group_data], ignore_index=True)
#         remaining = pd.DataFrame()  # 无剩余数据
#
#     # 剩余的数据放入 df_B
#     df_B = pd.concat([df_B, remaining], ignore_index=True)
#
#
#
# # 设定你自己的保存路径
# output_dir = "/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/COPD严重程度"
#
# # 确保目录存在
# os.makedirs(output_dir, exist_ok=True)
#
# # 保存抽取的数据到 Excel 文件
# df_A.to_excel(os.path.join(output_dir, "test重极重.xlsx"), index=False)
# df_B.to_excel(os.path.join(output_dir, "无需理会.xlsx"), index=False)
#
# print(f"数据已分配并保存到 {output_dir}Aexcel.xlsx 和 {output_dir}Bexcel.xlsx。")
#







# ###一个excel出现，不能在另一个出现
# import pandas as pd
#
#
# def remove_duplicates(first_excel, second_excel, output_excel):
#     # 读取第一个Excel的A列
#     df1 = pd.read_excel(first_excel, usecols=["A"], dtype=str)
#     filenames_to_remove = set(df1.iloc[:, 0].dropna())
#
#     # 读取第二个Excel的A列
#     df2 = pd.read_excel(second_excel, dtype=str)
#
#     # 过滤掉A列中出现在第一个Excel中的文件名
#     df_filtered = df2[~df2.iloc[:, 0].isin(filenames_to_remove)]
#
#     # 保存结果到新的Excel文件
#     df_filtered.to_excel(output_excel, index=False)
#     print(f"去重后数据已保存至: {output_excel}")
#
#
# # 示例调用
# remove_duplicates("/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_10_1/protect/分析现在74.xlsx",
#                   "/home/zsq/train/pre_process/DATE/符合分布的train和test/COPD_60_100.xlsx",
#                   "/home/zsq/train/MedicalNet_pytorch_files/trails/models/resnet_34_10_1/protect/74.xlsx")









# ########它读取第一个 Excel 文件的 A 列文件名，然后在第二个 Excel 文件中查找匹配的文件名，并提取整行数据
# import pandas as pd
#
# # 读取两个Excel文件
# file1 = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/重来吧/总的copd.xlsx'
# file2 = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/重来吧/12.xlsx'
#
# df1 = pd.read_excel(file1)
# df2 = pd.read_excel(file2)
#
# # 比较第一列，找到匹配的行
# matched_rows = df1[df1.iloc[:, 0].isin(df2.iloc[:, 0])]
#
# # 找到不匹配的行
# unmatched_rows_df1 = df1[~df1.iloc[:, 0].isin(df2.iloc[:, 0])]
# unmatched_rows_df2 = df2[~df2.iloc[:, 0].isin(df1.iloc[:, 0])]
#
# # 将匹配的行保存到一个新的Excel文件
# matched_rows.to_excel('/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/重来吧/新copd.xlsx', index=False)
#
# # 将不匹配的行保存到另一个Excel文件
# with pd.ExcelWriter('/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/重来吧/未匹配的copd.xlsx') as writer:
#     unmatched_rows_df1.to_excel(writer, sheet_name='Unmatched_File1', index=False)
#     unmatched_rows_df2.to_excel(writer, sheet_name='Unmatched_File2', index=False)
#
# print("处理完成，匹配的行已保存到 'matched_rows.xlsx'，不匹配的行已保存到 'unmatched_rows.xlsx'")








# ######映射
# import pandas as pd
# import numpy as np
# from scipy.stats import gaussian_kde
#
# # 读取Excel文件
# input_file = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/重来吧/G1＋G2+G3+G4/G1＋G2+G3+G4.xlsx'  # 输入文件名
# output_file = '/home/zsq/train/pre_process/DATE/符合分布的train和test/重新分男女/绘图/重来吧/G1＋G2+G3+G4/G1＋G2+G3+G4映射.xlsx'  # 输出文件名
#
# # 读取B列数据
# df = pd.read_excel(input_file, usecols=[4])  # B列是第2列，索引为1
# data = df.iloc[:, 0].dropna().values  # 提取B列数据并去除空值
#
# # 核密度估计
# kde = gaussian_kde(data)
# x_values = np.linspace(data.min(), data.max(), 1000)
# density = kde(x_values)
#
# # 计算CDF
# cdf = np.cumsum(density)
# cdf /= cdf[-1]  # 归一化
#
# # 自定义映射函数
# def custom_map(x, target_min, target_max):
#     """
#     将数据映射到自定义范围 [target_min, target_max]
#     :param x: 原始数据值
#     :param target_min: 目标范围的最小值
#     :param target_max: 目标范围的最大值
#     :return: 映射后的值
#     """
#     index = np.searchsorted(x_values, x)
#     cdf_value = cdf[index]  # 获取CDF值
#     return target_min + (target_max - target_min) * cdf_value  # 线性映射到自定义范围
#
# # 设置自定义映射范围
# target_min = 50  # 目标范围的最小值
# target_max = 100  # 目标范围的最大值
#
# # 对B列的每个值进行映射
# mapped_values = [custom_map(x, target_min, target_max) for x in data]
#
# # 将映射后的值写入新的Excel文件
# output_df = pd.DataFrame({'原始数据': data, '映射后的值': mapped_values})
# output_df.to_excel(output_file, index=False)
#
# print(f"映射完成，结果已保存到 {output_file}")












#########把不在A中不在B中的筛选出来
# import pandas as pd
#
# # 读取 ExcelA 的 A 列，假设列名为 '文件名'，从第一行开始
# excelA = pd.read_excel('/home/zsq/train/pre_process/DATE/外部验证excel/盛京/筛选人/第一次筛选.xlsx', usecols=[0], engine='openpyxl')
# file_names = excelA.iloc[:, 0].dropna().astype(str).tolist()
#
# # 读取 ExcelB
# excelB = pd.read_excel('/home/zsq/train/pre_process/DATE/外部验证excel/盛京/筛选人/总人数.xlsx', engine='openpyxl')
#
# # 假设检查检查号这一列名就是 'M'
# excelB['J'] = excelB['J'].astype(str)
#
# # 找出 excelB 中“检查检查号”不在 excelA 文件名中的行
# unmatched_rows = excelB[~excelB['J'].isin(file_names)]
#
# # 将这些行写入 ExcelC
# unmatched_rows.to_excel('/home/zsq/train/pre_process/DATE/外部验证excel/盛京/筛选人/z4.xlsx', index=False)
# print("未匹配的行已成功输出到 ExcelC.xlsx")









# import pandas as pd
#
# from openpyxl import load_workbook
# from openpyxl.workbook import Workbook
#
#
# def repair_excel(original_path):
#     """修复Excel文件并返回修复后的文件路径"""
#     try:
#         # 创建修复后的文件路径
#         dir_name = os.path.dirname(original_path)
#         base_name = os.path.basename(original_path)
#         repaired_path = os.path.join(dir_name, f"repaired_{base_name}")
#
#         # 读取原始文件并创建新工作簿
#         wb = load_workbook(original_path, read_only=True)
#         new_wb = Workbook()
#
#         # 删除默认创建的空工作表
#         if new_wb.sheetnames:
#             del new_wb[new_wb.sheetnames[0]]
#
#         # 复制所有工作表和数据
#         for sheet_name in wb.sheetnames:
#             new_sheet = new_wb.create_sheet(sheet_name)
#             sheet = wb[sheet_name]
#
#             # 复制单元格数据
#             for row in sheet.iter_rows():
#                 new_sheet.append([cell.value for cell in row])
#
#         # 保存修复后的文件
#         new_wb.save(repaired_path)
#         return repaired_path
#
#     except Exception as e:
#         print(f"修复失败: {str(e)}")
#         return None
#
#
# # 读取Excel文件
# file_path = "/home/zsq/train/pre_process/DATE/外部验证excel/其余关系/算均值±方差/11.xlsx"  # 替换为你的Excel文件路径
# file_path= repair_excel(file_path)
#
# df = pd.read_excel(file_path)
#
# # # 选择Q列、S列、T列（列名可能是字母或数字，需调整）
# # columns_to_analyze = ["Q", "S", "T"]  # 如果列名是字母
# # 或者用数字索引（如第17列是Q列，第19列是S列，第20列是T列）：
# columns_to_analyze = df.iloc[:, [16, 18, 19, 20]]  # 索引从0开始
#
# # 计算均值和方差
# results = {}
# for col in columns_to_analyze:
#     data = df[col]     # .dropna()  # 去除NaN值
#     mean = data.mean()
#     std = data.std()
#     results[col] = {"Mean": mean, "std": std}
#
# # 打印结果
# for col, stats in results.items():
#     print(f"列 {col}:")
#     if(col == "all lung em index"):
#         print(f"  均值 = {stats['Mean']:.3f}")
#         print(f"  标准差 = {stats['std']:.3f}")
#         print("-" * 30)
#     else:
#         print(f"  均值 = {stats['Mean']:.1f}")
#         print(f"  标准差 = {stats['std']:.1f}")
#         print("-" * 30)

# # 可选：保存结果到新Excel文件
# output_df = pd.DataFrame.from_dict(results, orient="index")
# output_df.to_excel("output_statistics.xlsx", index_label="Column")
# print("结果已保存到 output_statistics.xlsx")











##########npy 2 nii
# import numpy as np
# import SimpleITK as sitk
# import os
# import traceback
#
#
# def npy_to_nii(npy_path, origin=None, spacing=None, direction=None):
#     """
#     将npy文件转换为nii.gz格式
#
#     参数:
#         npy_path: npy文件路径
#         origin: 原始图像原点坐标(optional)
#         spacing: 原始图像体素间距(optional)
#         direction: 原始图像方向矩阵(optional)
#
#     返回:
#         sitk图像对象
#     """
#     # 1. 加载npy数据
#     img_array = np.load(npy_path)
#
#     # 2. 数据验证
#     if not isinstance(img_array, np.ndarray):
#         raise ValueError("Loaded object is not a numpy array")
#
#     if img_array.size == 0:
#         raise RuntimeError("Loaded numpy array is empty")
#
#     # 3. 转换为SimpleITK图像
#     itk_img = sitk.GetImageFromArray(img_array)
#
#     # 4. 设置空间信息(优先使用提供的参数)
#     if origin is not None:
#         itk_img.SetOrigin(origin)
#     # elif 'original_origin' in locals():
#         # itk_img.SetOrigin(original_origin)
#
#     if spacing is not None:
#         itk_img.SetSpacing(spacing)
#     # elif 'original_spacing' in locals():
#     #     itk_img.SetSpacing(original_spacing)
#
#     if direction is not None:
#         itk_img.SetDirection(direction)
#     # elif 'original_direction' in locals():
#     #     itk_img.SetDirection(original_direction)
#
#     # 5. 添加默认空间信息
#     if itk_img.GetOrigin() == (0.0, 0.0, 0.0):
#         print(f"警告: 使用默认原点 (0,0,0) 替代缺失信息")
#
#     if itk_img.GetSpacing() == (1.0, 1.0, 1.0):
#         print(f"警告: 使用默认间距 (1,1,1) 替代缺失信息")
#
#     return itk_img
#
#
# def convert_npy_folder_to_nii(input_folder, output_folder, reference_nii=None):
#     """
#     批量转换文件夹中的npy文件到nii.gz
#
#     参数:
#         input_folder: 包含npy文件的输入目录
#         output_folder: 输出nii.gz文件的目录
#         reference_nii: 提供空间信息的参考nii文件(可选)
#     """
#     # 准备参考信息
#     if reference_nii and os.path.exists(reference_nii):
#         ref_img = sitk.ReadImage(reference_nii)
#         origin = ref_img.GetOrigin()
#         spacing = ref_img.GetSpacing()
#         direction = ref_img.GetDirection()
#     else:
#         origin = spacing = direction = None
#
#     # 确保输出目录存在
#     os.makedirs(output_folder, exist_ok=True)
#
#     # 遍历npy文件
#     conversion_count = 0
#     error_files = []
#
#     for file in os.listdir(input_folder):
#         if file.endswith('.npy'):
#             try:
#                 npy_path = os.path.join(input_folder, file)
#
#                 # 生成输出路径
#                 nii_filename = file.replace('.npy', '.nii.gz')
#                 nii_path = os.path.join(output_folder, nii_filename)
#
#                 # 转换并保存
#                 itk_img = npy_to_nii(npy_path, origin, spacing, direction)
#                 sitk.WriteImage(itk_img, nii_path)
#                 conversion_count += 1
#
#                 print(f"成功转换: {file} -> {nii_filename}")
#
#             except Exception as e:
#                 print(f"转换失败: {file}")
#                 print(traceback.format_exc())
#                 error_files.append(file)
#
#     # 结果总结
#     print(f"\n转换总结:")
#     print(f"总尝试转换: {len(os.listdir(input_folder))}")
#     print(f"成功转换: {conversion_count}")
#     print(f"失败: {len(error_files)}")
#
#     if error_files:
#         print("问题文件列表:")
#         for file in error_files:
#             print(f" - {file}")
#
#
# if __name__ == "__main__":
#     # 配置路径
#     input_folder = "/home/zsq/train/pre_process/npy2nii"
#     output_folder = "/home/zsq/train/pre_process/npy2nii"
#     reference_nii = "/home/zsq/train/pre_process/train/A03-2021-11-19-021111100055-chensuo_nnnnn_20220612005730787981-SE203_2583_0000.nii.gz"  # 可选参考文件
#
#     # 执行批量转换
#     convert_npy_folder_to_nii(input_folder, output_folder, reference_nii)