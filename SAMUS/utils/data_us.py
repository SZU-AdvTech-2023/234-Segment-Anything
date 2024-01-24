import os
from random import randint
import numpy as np
import torch
from skimage import io, color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
import pandas as pd
from numbers import Number
from typing import Container
from collections import defaultdict
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from einops import rearrange
import random
import albumentations as A
import torchvision.transforms as transforms
from PIL import Image
from random import randint
import random

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

def sort_by_number(line):
    num = []
    # 分割每一行，提取其中的数字部分
    for i in line.split('.'):
        if i.isdigit():
            num.append(int(i))
    return num[0]  # 返回第一个数字

def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def random_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], [point_label]

def fixed_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[len(indices)//2] 
    return pt[np.newaxis, :], [point_label]


def random_clicks(mask, class_id = 1, prompts_number=10):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt_index = np.random.randint(len(indices), size=prompts_number)
    pt = indices[pt_index]
    point_label = np.repeat(point_label, prompts_number)
    return pt, point_label

def pos_neg_clicks(mask, class_id=1, pos_prompt_number=5, neg_prompt_number=5):
    pos_indices = np.argwhere(mask == class_id)
    pos_indices[:, [0,1]] = pos_indices[:, [1,0]]
    pos_prompt_indices = np.random.randint(len(pos_indices), size=pos_prompt_number)
    pos_prompt = pos_indices[pos_prompt_indices]
    pos_label = np.repeat(1, pos_prompt_number)

    neg_indices = np.argwhere(mask != class_id)
    neg_indices[:, [0,1]] = neg_indices[:, [1,0]]
    neg_prompt_indices = np.random.randint(len(neg_indices), size=neg_prompt_number)
    neg_prompt = neg_indices[neg_prompt_indices]
    neg_label = np.repeat(0, neg_prompt_number)

    pt = np.vstack((pos_prompt, neg_prompt))
    point_label = np.hstack((pos_label, neg_label))
    return pt, point_label

def random_bbox(mask, class_id=1, img_size=256):
    # return box = np.array([x1, y1, x2, y2])
    indices = np.argwhere(mask == class_id) # Y X
    indices[:, [0,1]] = indices[:, [1,0]] # x, y
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])

    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])

    classw_size = maxx-minx+1
    classh_size = maxy-miny+1

    shiftw = randint(int(0.95*classw_size), int(1.05*classw_size))
    shifth = randint(int(0.95*classh_size), int(1.05*classh_size))
    shiftx = randint(-int(0.05*classw_size), int(0.05*classw_size))
    shifty = randint(-int(0.05*classh_size), int(0.05*classh_size))

    new_centerx = (minx + maxx)//2 + shiftx
    new_centery = (miny + maxy)//2 + shifty

    minx = np.max([new_centerx-shiftw//2, 0])
    maxx = np.min([new_centerx+shiftw//2, img_size-1])
    miny = np.max([new_centery-shifth//2, 0])
    maxy = np.min([new_centery+shifth//2, img_size-1])

    return np.array([minx, miny, maxx, maxy])

def fixed_bbox(mask, class_id = 1, img_size=256):
    indices = np.argwhere(mask == class_id) # Y X (0, 1)
    indices[:, [0,1]] = indices[:, [1,0]]
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])
    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])
    return np.array([minx, miny, maxx, maxy])

def visual_segmentation_sets_with_pt(seg, image_filename, save, pt):
    img_ori = cv2.imread(image_filename)
    img_ori0 = cv2.imread(image_filename)
    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                      [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])

    # seg0 = seg[0, :, :]

    for i in range(1, 2):
        img_r[seg == i] = table[i - 1, 0]
        img_g[seg == i] = table[i - 1, 1]
        img_b[seg == i] = table[i - 1, 2]

    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)

    img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0)
    # img = img_ori0

    # pt = np.array(pt.cpu())
    N = pt.shape[0]
    for i in range(N):
        cv2.circle(img, (int(pt[i, 0]), int(pt[i, 1])), 6, (0,0,0), -1)
        cv2.circle(img, (int(pt[i, 0]), int(pt[i, 1])), 5, (0,0,255), -1)
        cv2.line(img, (int(pt[i, 0]-3), int(pt[i, 1])), (int(pt[i, 0])+3, int(pt[i, 1])), (0, 0, 0), 1)
        cv2.line(img, (int(pt[i, 0]), int(pt[i, 1])-3), (int(pt[i, 0]), int(pt[i, 1])+3), (0, 0, 0), 1)


    cv2.imwrite(save, img)
class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, img_size=256, low_img_size=256, ori_size=256, crop=(32, 32), p_flip=0.0, p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.low_img_size = low_img_size
        self.ori_size = ori_size


    def __call__(self, image, mask):
        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image, mask = F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR), F.resize(mask, (new_h, new_w), InterpolationMode.NEAREST)
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)
        # transforming to tensor
        image, mask = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR), F.resize(mask, (self.ori_size, self.ori_size), InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size), InterpolationMode.NEAREST)
        image = F.to_tensor(image)

        if not self.long_mask:
            mask = F.to_tensor(mask)
            low_mask = F.to_tensor(low_mask)
        else:
            mask = to_long_tensor(mask)
            low_mask = to_long_tensor(low_mask)
        # print('JointTransform2D-----OK')
        # print("----------------------------------------")
        # mask = torch.unsqueeze(mask,dim=0)
        # low_mask = torch.unsqueeze(low_mask,dim=0)
        # print("image:",type(image))
        # print("mask:",type(mask))
        # print("low_mask:",type(low_mask))
        # print("image:", image.shape)  #(1,256,256)
        # print("mask:", mask.shape)       #(256,256)
        # print("low_mask:", low_mask.shape)    #(128,128)
        # image_pil = transforms.ToPILImage()(image[0].cpu())
        # image_pil.save(os.path.join("/data/gjx/project/SAMUS/SAMUS/JointTransform2D/image", "image.png"))
        # mask_pil = F.to_pil_image(mask)
        # mask_pil.save(os.path.join("/data/gjx/project/SAMUS/SAMUS/JointTransform2D/mask", "mask.png"))
        # low_mask_pil = transforms.ToPILImage()(low_mask[0].cpu())
        # low_mask_pil.save(os.path.join("/data/gjx/project/SAMUS/SAMUS/JointTransform2D/low_mask", "low_mask.png"))
        # print("----------------------------------------")
        return image, mask, low_mask


# class ImageToImage2D(Dataset):
#     """
#     Reads the images and applies the augmentation transform on them.
#
#     Args:
#         dataset_path: path to the dataset. Structure of the dataset should be:
#             dataset_path
#                 |-- MainPatient
#                     |-- train.txt
#                     |-- val.txt
#                     |-- text.txt
#                         {subtaski}/{imgname}
#                     |-- class.json
#                 |-- subtask1
#                     |-- img
#                         |-- img001.png
#                         |-- img002.png
#                         |-- ...
#                     |-- label
#                         |-- img001.png
#                         |-- img002.png
#                         |-- ...
#                 |-- subtask2
#                     |-- img
#                         |-- img001.png
#                         |-- img002.png
#                         |-- ...
#                     |-- label
#                         |-- img001.png
#                         |-- img002.png
#                         |-- ...
#                 |-- subtask...
#
#         joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
#             evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
#         one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
#     """
#
#     def __init__(self, dataset_path: str, split='train', joint_transform: Callable = None, img_size=256, prompt = "click", class_id=1,
#                  one_hot_mask: int = False) -> None:
#         self.dataset_path = dataset_path
#         self.one_hot_mask = one_hot_mask
#         self.split = split
#         id_list_file = os.path.join(dataset_path, 'MainPatient/{0}.txt'.format(split))
#         self.ids = [id_.strip() for id_ in open(id_list_file)]
#         self.prompt = prompt
#         self.img_size = img_size
#         self.class_id = class_id
#         self.class_dict_file = os.path.join(dataset_path, 'MainPatient/class.json')
#         with open(self.class_dict_file, 'r') as load_f:
#             self.class_dict = json.load(load_f)
#         if joint_transform:
#             self.joint_transform = joint_transform
#         else:
#             to_tensor = T.ToTensor()
#             self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
#
#     def __len__(self):
#         return len(self.ids)
#
#     def __getitem__(self, i):
#         id_ = self.ids[i]
#         if "test" in self.split:
#             sub_path, filename = id_.split('/')[0], id_.split('/')[1]
#             # class_id0, sub_path, filename = id_.split('/')[0], id_.split('/')[1], id_.split('/')[2]
#             # self.class_id = int(class_id0)
#         else:
#             class_id0, sub_path, filename = id_.split('/')[0], id_.split('/')[1], id_.split('/')[2]
#         img_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'img')
#         label_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'label')
#         image = cv2.imread(os.path.join(img_path, filename + '.png'), 0)
#         mask = cv2.imread(os.path.join(label_path, filename + '.png'), 0)
#         classes = self.class_dict[sub_path]
#         if classes == 2:
#             mask[mask > 1] = 1
#
#         # correct dimensions if needed
#         image, mask = correct_dims(image, mask)
#         if self.joint_transform:
#             image, mask, low_mask = self.joint_transform(image, mask)
#         if self.one_hot_mask:
#             assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
#             mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
#
#          # --------- make the point prompt -----------------
#         if self.prompt == 'click':
#             point_label = 1
#             if 'train' in self.split:
#                 #class_id = randint(1, classes-1)
#                 class_id = int(class_id0)
#             elif 'val' in self.split:
#                 class_id = int(class_id0)
#             else:
#                 class_id = self.class_id
#             if 'train' in self.split:
#                 pt, point_label = random_click(np.array(mask), class_id)
#                 bbox = random_bbox(np.array(mask), class_id, self.img_size)
#             else:
#                 pt, point_label = fixed_click(np.array(mask), class_id)
#                 bbox = fixed_bbox(np.array(mask), class_id, self.img_size)
#             mask[mask!=class_id] = 0
#             mask[mask==class_id] = 1
#             low_mask[low_mask!=class_id] = 0
#             low_mask[low_mask==class_id] = 1
#             point_labels = np.array(point_label)
#         if self.one_hot_mask:
#             assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
#             mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
#
#         low_mask = low_mask.unsqueeze(0)
#         mask = mask.unsqueeze(0)
#         return {
#             'image': image,
#             'label': mask,
#             'p_label': point_labels,
#             'pt': pt,
#             'bbox': bbox,
#             'low_mask':low_mask,
#             'image_name': filename + '.png',
#             'class_id': class_id,
#             }
class ImageToImage2D(Dataset):
    def __init__(self, filepath, joint_transform, mode):
        self.filepath = filepath
        self.mode = mode
        self.joint_transform = joint_transform

        # self.train_transform = A.Compose([
        #     A.RandomContrast(limit=0.2, p=1),  # 对比度增强
        #     A.RandomBrightness(limit=0.2, p=1),  # 亮度增强
        #     A.Rotate(limit=30, p=0.5),  # 随机旋转
        #     A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # 模糊
        # ])
        if self.mode == 'train':
            self.data_path = (os.path.join(os.path.join(filepath, 'train'), 'img'))
            self.mask_path = (os.path.join(os.path.join(filepath, 'train'), 'label'))
        # elif self.mode == 'val':
        #     self.data_path = (os.path.join(os.path.join(filepath, 'val'), 'img'))
        #     self.mask_path = (os.path.join(os.path.join(filepath, 'val'), 'mask'))
        elif self.mode == 'test':
            self.data_path = (os.path.join(os.path.join(filepath, 'test'), 'img'))
            self.mask_path = (os.path.join(os.path.join(filepath, 'test'), 'label'))

        self.data = os.listdir(self.data_path)
        self.mask = os.listdir(self.mask_path)

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = self.data[idx]
        # print("id",id)
        patient_image = os.path.join(self.data_path, id)
        patient_label = os.path.join(self.mask_path, id)
        img_name = os.listdir(patient_image)
        mask_name = os.listdir(patient_label)
        img_name.sort(key=sort_by_number)
        mask_name.sort(key=sort_by_number)
        # for image_file in img_name:
        #     print("image_file",image_file)
        num = len(img_name)
        # print("len",num)
        self.size = (256, 256, num)
        self.low_size = (128, 128, num)

        img_data = np.zeros(self.size, dtype=np.float32)
        mask_data = np.zeros(self.size, dtype=np.float32)
        low_mask_data = np.zeros(self.low_size, dtype=np.float32)

        sum_pt = []
        sum_point_label = []

        sum_img_path = []
        sum_img_name = []
        for k,(i,j) in enumerate(zip(img_name, mask_name)):
            # print("i",i)
            temp_img = os.path.join(patient_image, i)
            temp_mask = os.path.join(patient_label, j)
            # print("temp_img",temp_img)
            image = cv2.imread(temp_img, 0)
            mask = cv2.imread(temp_mask, 0)
            # print("image.shape",image.shape)  (256,256)
            # print("temp",temp_img)
            # print("mask.shape",mask.shape)    (256,256)
            # elements_greater_than_one = mask[mask > 1]
            #
            # # 输出大于1的元素数量
            # print("Number of elements greater than 1:", len(elements_greater_than_one))
            image, mask = correct_dims(image, mask)
            # print("image.shape",image.shape)   #(256,256,1)
            # print("mask.shape",mask.shape)     #(256,256,1)
            if self.joint_transform:
                 image, mask, low_mask = self.joint_transform(image, mask)

            # print("low_mask.shape", low_mask.shape)   #(128,128)
            # print("image.shape", image.shape)         #(1,256,256)
            # print("mask.shape", mask.shape)           #(256,256)
            # np.set_printoptions(threshold=np.inf)
            # print("mask",mask)
            # elements_greater_than_one = mask[mask == 1]
            #
            # # 输出大于1的元素数量
            # print("Number of elements greater than 1:", len(elements_greater_than_one))
            mask[mask == 1] = 1
            mask[mask != 1] = 0
            # mask = np.array(mask).astype(np.uint8)
            temp1 = np.array(mask)
            temp1 = temp1.astype(np.uint8)
            temp1[temp1 == 1] = 255
            # count = np.sum(temp1 == 255)
            # print("矩阵中大于1的元素个数:", count)
            # tei = Image.fromarray(temp1)
            # if self.mode == 'train':
            #     ppp = "/data/gjx/project/SAMUS/SAMUS/result/train_point/" + re_img
            # else:
            #     ppp = "/data/gjx/project/SAMUS/SAMUS/result/test_point/" + re_img
            # if not os.path.exists(ppp):
            #     os.mkdir(ppp)
            # tei.save(os.path.join(ppp,re_img))
            low_mask[low_mask == 1] = 1
            low_mask[low_mask != 1] = 0



            if self.mode == 'train':
                pt, point_label = random_click(np.array(mask), class_id=1)
                bbox = random_bbox(np.array(mask), class_id=1)
            else:
                pt, point_label = fixed_click(np.array(mask), class_id=1)
                bbox = fixed_bbox(np.array(mask), class_id=1, img_size=256)
            # print("pt",pt.shape)
            # mask[mask != 1] = 0
            # mask[mask == 1] = 1
            if self.mode == 'train':
                low_mask[low_mask != 1] = 0
                low_mask[low_mask == 1] = 1
            low_mask = low_mask.unsqueeze(0)
            point_labels = np.array(point_label)
            # visual_segmentation_sets_with_pt(temp1,temp_img,ppp,pt)
            # print("pt.shape", pt.shape)
            # print("point_labels.shape", point_labels.shape)
            mask = mask.unsqueeze(0)
            # print("low_mask.shape", low_mask.shape)  (1,128,128)
            # new_re_img = re_img[7:]  #去除patient
            # print("new_re_img", new_re_img)
            sum_pt.append(pt)
            sum_point_label.append(point_labels)

            # mask = mask.unsqueeze(0)
            image = image.squeeze(0)
            # print("image",image.shape)
            # print("image mask low_mask", image.shape,mask.shape, low_mask.shape)
            img_data[..., k] = image.reshape((self.size[0], self.size[1]))
            mask_data[..., k] = mask
            low_mask_data[..., k] = low_mask

        sum_pt = np.array(sum_pt)
        # print("sum_pt",type(sum_pt))
        sum_point_label = np.array(sum_point_label)

        final_img = np.expand_dims(img_data, axis=0).transpose(3, 0, 1, 2)    #b c h w
        final_mask = np.expand_dims(mask_data, axis=0).transpose(3, 0, 1, 2)
        final_low_mask = np.expand_dims(low_mask_data, axis=0).transpose(3, 0, 1, 2)
        # print("final_img",final_img.shape)
        # print("final_mask", final_mask.shape)
        # print("final_low_mask", final_low_mask.shape)
        return {
            'image': final_img,
            'label': final_mask,
            'p_label': sum_point_label,
            'pt': sum_pt,
            # 'bbox': bbox,
            'low_mask': final_low_mask,
            'image_name': sum_img_name,
            'class_id': 1,
            'num':num,
            'patient_name': id
        }

class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


