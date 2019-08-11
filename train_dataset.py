from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
import os
import cv2
from os.path import isfile
import numpy as np
from config import Config


# The Code from: https://www.kaggle.com/ratthachat/aptos-updated-albumentation-meets-grad-cam
cfg = Config()

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


class CreateDataset(Dataset):

    def __init__(self, df_data, data_dir, transform=None):
        self.df = df_data
        self.transform = transform
        self.train_path = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.diagnosis.values[idx]
        label = np.expand_dims(label, -1)

        p = self.df.id_code.values[idx]
        p_path = expand_path(p, self.train_path)
        image = cv2.imread(p_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = crop_image_from_gray(image)
        # image = cv2.resize(image, (cfg.img_size, cfg.img_size))
        # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label

transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-120, 120)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transforms_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])