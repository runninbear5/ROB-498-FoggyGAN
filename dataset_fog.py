import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import cv2
import os
import numpy as np
from options import opt
import torchvision
import torchvision.transforms.functional as F
import numbers
import random
from PIL import Image
import glob
from typing import Tuple

class ToTensor(object):
    def __call__(self, sample):
        hazy_image, clean_image = sample['hazy'], sample['clean']
        hazy_image = torch.from_numpy(np.array(hazy_image).astype(np.float32))
        hazy_image = torch.transpose(torch.transpose(hazy_image, 2, 0), 1, 2)
        clean_image = torch.from_numpy(np.array(clean_image).astype(np.float32))
        clean_image = torch.transpose(torch.transpose(clean_image, 2, 0), 1, 2)
        return {'hazy': hazy_image,
                'clean': clean_image}

def resize_with_pad(image: np.array,
                    new_shape: Tuple[int, int],
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image


class Dataset_Load(Dataset):
    def __init__(self, data_path, gt_path, transform=None):
        self.data_path = data_path
        self.gt_path = gt_path
        self.filesA, self.filesB = self.get_file_paths(self.data_path, self.gt_path)
        self.len = min(len(self.filesA), len(self.filesB))
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        hazy_im = resize_with_pad(cv2.imread(self.filesA[index % self.len]), (512,512),
                                 padding_color=(0,0,0))

        hazy_im = hazy_im[:, :, ::-1] ## BGR to RGB
        hazy_im = np.float32(hazy_im) / 255.0


        clean_im = resize_with_pad(cv2.imread(self.filesB[index % self.len]), (512,512),
                                  padding_color=(0,0,0))

        clean_im = clean_im[:, :, ::-1] ## BGR to RGB
        clean_im = np.float32(clean_im) / 255.0

        sample = {'hazy': hazy_im,
                  'clean': clean_im}
        if self.transform != None:
            sample = self.transform(sample)

        return sample


    def get_file_paths(self, root, gt_path):
        filesA, filesB = [], []

        for city in os.listdir(root):
            filesA += sorted([x for x in glob.glob(os.path.join(root, city) + "/*.*") if "0.02" in x])
            filesB += sorted(glob.glob(os.path.join(gt_path, city) + "/*.*"))

        return filesA, filesB
