from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
import os
import cv2
from os.path import isfile
import numpy as np
from config import Config
import math

from utils_img import *
from retinal_process import *

# The Code from: https://www.kaggle.com/ratthachat/aptos-updated-albumentation-meets-grad-cam
cfg = Config()
# import tensorflow as tf

def crop_image1(img, tol=7):
    # img is image data
    # tol  is tolerance

    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img

def expand_path(p, train_path):
    p = str(p)
    if isfile(train_path + p + ".png"):
        return train_path + (p + ".png")
    # if isfile(train_old_path + p + '.png'):
    #     return train_old_path + (p + ".png")
    # if isfile(test + p + ".png"):
    #     return test + (p + ".png")
    return p



def subtract_median_bg_image(im):
    k = np.max(im.shape)//20*2+1
    bg = cv2.medianBlur(im, k)
    return cv2.addWeighted (im, 4, bg, -4, 128)

PARAM = 96
def Radius_Reduction(img,PARAM):
    h,w,c=img.shape
    Frame=np.zeros((h,w,c),dtype=np.uint8)
    cv2.circle(Frame,(int(math.floor(w/2)),int(math.floor(h/2))),int(math.floor((h*PARAM)/float(2*100))), (255,255,255), -1)
    Frame1=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    img1 =cv2.bitwise_and(img,img,mask=Frame1)
    return img1

def info_image(im):
    # Compute the center (cx, cy) and radius of the eye
    cy = im.shape[0]//2
    midline = im[cy,:]
    midline = np.where(midline>midline.mean()/3)[0]
    if len(midline)>im.shape[1]//2:
        x_start, x_end = np.min(midline), np.max(midline)
    else: # This actually rarely happens p~1/10000
        x_start, x_end = im.shape[1]//10, 9*im.shape[1]//10
    cx = (x_start + x_end)/2
    r = (x_end - x_start)/2
    return cx, cy, r

def resize_image(im, image_size, augmentation=False):
    # Crops, resizes and potentially augments the image to IMAGE_SIZE
    cx, cy, r = info_image(im)
    scaling = image_size/(2*r)
    rotation = 0
    if augmentation:
        scaling *= 1 + 0.3 * (np.random.rand()-0.5)
        rotation = 360 * np.random.rand()
    M = cv2.getRotationMatrix2D((cx,cy), rotation, scaling)
    M[0,2] -= cx - image_size/2
    M[1,2] -= cy - image_size/2
    return cv2.warpAffine(im,M,(image_size, image_size))

class CreateDataset(Dataset):

    def __init__(self, df_data, data_dir, transform=None):
        self.df = df_data
        self.transform = transform
        self.train_path = data_dir
        self.number = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.diagnosis.values[idx]
        label = np.expand_dims(label, -1)

        p = self.df.id_code.values[idx]
        p_path = expand_path(p, self.train_path)
        image = cv2.imread(p_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # lab_planes = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        # lab_planes[0] = clahe.apply(lab_planes[0])
        # lab = cv2.merge(lab_planes)
        # clahed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # image = resize_image(clahed, cfg.img_size)
        # print(tf.test.is_gpu_available())
        # vessel_model = VesselNet('./vessels/')
        image = process(image, size = cfg.img_size, crop='normal', preprocessing='clahe', fourth=None)
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label

transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-150, 150)),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.5, saturation=0.1, hue=0.1),
        # transforms.RandomResizedCrop(cfg.img_size_crop),
        transforms.ToTensor(),
        transforms.Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229]),
        ])

transforms_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229])
    ])