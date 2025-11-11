import SimpleITK as sitk
import os
import numpy as np
import tqdm
import copy
import cv2

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
def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list
def save_itk(image, origin, spacing, filename):
    """
    :param image: images to be saved
    :param origin: CT origin
    :param spacing: CT spacing
    :param filename: save name
    :return: None
    """
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)

def lumTrans_hu(img):
	"""
	:param img: CT image
	:return: Hounsfield Unit window clipped and normalized
	"""
	lungwin = np.array([-1000.,400.])
	newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
	newimg[newimg < 0] = 0
	newimg[newimg > 1] = 1
	newimg = (newimg*255).astype('uint8')
	return newimg

def crop_roi(ct,lungmask):
    lungmask_img = sitk.ReadImage(lungmask)
    Mask = sitk.GetArrayFromImage(lungmask_img)
    xx, yy, zz = np.where(Mask)

    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])

    box = np.vstack([np.max([[0, 0, 0], box[:, 0]-5 ], 0), np.min([np.array(Mask.shape), box[:, 1]+5 ], axis=0).T]).T
    ct_img = sitk.ReadImage(ct)
    ct = sitk.GetArrayFromImage(ct_img)
    ct2_img= ct[box[0, 0]:box[0, 1],box[1, 0]:box[1, 1],box[2, 0]:box[2, 1]]
    return  ct2_img
def boxx(ct,lungmask):
    (filepath, tempfilename)=os.path.split(ct)
    (filename, extension) = os.path.splitext(tempfilename)
    (filename, extension) = os.path.splitext(filename)
    lungmask_img = sitk.ReadImage(lungmask)
    Mask = sitk.GetArrayFromImage(lungmask_img)
    xx, yy, zz = np.where(Mask)

    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])

    box = np.vstack([np.max([[0, 0, 0], box[:, 0] - 5], 0), np.min([np.array(Mask.shape), box[:, 1] + 5], axis=0).T]).T
    ct_img = sitk.ReadImage(ct)
    ct = sitk.GetArrayFromImage(ct_img)

    shapeorg = ct.shape
    box_shape = np.array([[0, shapeorg[0]], [0, shapeorg[1]], [0, shapeorg[2]]])
    box = np.concatenate([box, box_shape], axis=0)
    return box

def savenpy(data_path, lung_mask, prep_folder):
    """
    :param data_path: input CT data path
    :param prep_folder:
    :return: None
    """
    resolution = np.array([1, 1, 1])
    (filepath, tempfilename) = os.path.split(data_path)
    (filename, extension) = os.path.splitext(tempfilename)
    (filename, extension) = os.path.splitext(filename)
    im = sitk.ReadImage(data_path)
    img = sitk.GetArrayFromImage(im)
    mask = sitk.ReadImage(lung_mask)
    Mask = sitk.GetArrayFromImage(mask)
    print(img.shape, end=" ")
    # im, m1, m2, mtotal, origin, spacing = step1_python(data_path)
    # print('Origin: ', origin, ' Spacing: ', spacing, 'img shape: ', im.shape)
    # Mask = m1 + m2
    xx, yy, zz = np.where(Mask)
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    margin = 0

    box = np.vstack(
        [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([np.array(Mask.shape), box[:, 1] + margin], axis=0).T]).T

    # save the lung mask
    img[np.isnan(img)] = -2000
    sliceim_hu = lumTrans_hu(Mask)
    shapeorg = sliceim_hu.shape
    box_shape = np.array([[0, shapeorg[0]], [0, shapeorg[1]], [0, shapeorg[2]]])


    Mask_crop = Mask[box_shape[0, 0]:box_shape[0, 1],
                box[1, 0]:box[1, 1],
                box[2, 0]:box[2, 1]]

    # save_itk(Mask_crop ,im.GetOrigin(), im.GetSpacing(), os.path.join(prep_folder, filename + "_lungmask_hu.nii.gz"))
    # convex_mask = m1
    # dm1 = process_mask(m1)
    # dm2 = process_mask(m2)
    # dilatedMask = (dm1 + dm2) | mtotal
    # Mask = m1+m2
    # dilatedMask = dilatedMask.astype('uint8')
    # Mask = Mask.astype('uint8')

    # save the CT image
    sliceim2_hu = img[box[0, 0]:box[0, 1],
                  box[1, 0]:box[1, 1],
                  box[2, 0]:box[2, 1]]
    box[0, 1] = box_shape[0, 1]
    box[0, 0] = box_shape[0, 0]
    # save box (image original shape and cropped window region)
    box = np.concatenate([box, box_shape], axis=0)
    # np.save(os.path.join(prep_folder, filename + '_box.npy'), box)
    # save processed image
    print(" -> ", sliceim2_hu.shape, end=" ")

    save_itk(sliceim2_hu, im.GetOrigin(), im.GetSpacing(),os.path.join(prep_folder, filename + ".nii.gz"))
    return


if __name__ == '__main__':
    # crop image path

    #img_path = '/disk2/wuyanan/wanren/identification/SSL/data'
    img_path =r'E:\CT\training(age)\0'
    # crop image path
    # label_path ='/disk2/wuyanan/wanren/identification/training/data/0'
    # the lobe or lung mask path for cropping image and label

    #lung_mask_path = '/disk2/wuyanan/wanren/identification/SSL/lobe'
    lung_mask_path =r'E:\CT\training(age)\xin_mask'
    # save path
    #save_path = '/disk1/wuyanan/wanren/identification/lung_box/ssl'
    save_path =r'E:\CT\training(age)\ce'

    img_path = get_listdir(img_path)
    img_path.sort()
    # label_path = get_listdir(label_path)
    # label_path.sort()
    l_mask = get_listdir(lung_mask_path)
    l_mask.sort()

    for i in tqdm.trange(len(img_path)):#
        (filepath, tempfilename) = os.path.split(img_path[i])
        (filename, extension) = os.path.splitext(tempfilename)
        (filename, extension) = os.path.splitext(filename)
        savenpy(img_path[i], l_mask[i], save_path)

