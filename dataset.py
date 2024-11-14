from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import numpy as np
from imresize import *
from torchvision import transforms
import os
import pickle

# 데이터셋 클래스 정의
class LoadDataset(Dataset):
    def __init__(self, hr_images, scaling_factor, transform=None):
        self.hr_images = hr_images
        self.scaling_factor = scaling_factor
        self.transform = transform

    def __len__(self):
        return len(self.hr_images)

    def center_crop(self, hr_img, scaling_factor):
        height, width = hr_img.shape[0:2]

        # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
        x_remainder = width % (
            2 * scaling_factor if scaling_factor == 1.5 else scaling_factor
        )
        y_remainder = height % (
            2 * scaling_factor if scaling_factor == 1.5 else scaling_factor
        )
        left = int(x_remainder // 2)
        top = int(y_remainder // 2)
        right = int(left + (width - x_remainder))
        bottom = int(top + (height - y_remainder))
        hr_img = hr_img[top:bottom, left:right]
        return hr_img

    def __getitem__(self, idx):
        hr_img_path = self.hr_images[idx]
        
        if bool(self.transform):
            hr_img = Image.open(hr_img_path).convert('RGB')
            hr_img = np.array(hr_img, dtype="float64")
            hr_img = self.transform(image=hr_img)
            hr_img = hr_img['image']
        else : 
            hr_img = cv2.imread(hr_img_path)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            hr_img = self.center_crop(hr_img, self.scaling_factor)
            hr_img = np.array(hr_img, dtype="float64")

        hr_img = self.center_crop(hr_img, self.scaling_factor)
        
        hr_height, hr_width = hr_img.shape[0:2]
        lr_img = imresize(hr_img, 1.0 / self.scaling_factor)  # equivalent to matlab's imresize
        lr_img = np.uint8(
            np.clip(lr_img, 0.0, 255.0)
        )  # this is to simulate matlab's imwrite operation
        hr_img = np.uint8(hr_img)

        lr_height, lr_width = lr_img.shape[0:2]
        # Sanity check
        assert (
            hr_width == lr_width * self.scaling_factor
            and hr_height == lr_height * self.scaling_factor
        )

        lr_img = torch.from_numpy(lr_img.transpose((2, 0, 1))).contiguous()
        lr_img = lr_img.to(dtype=torch.float32).div(255)

        hr_img = torch.from_numpy(hr_img.transpose((2, 0, 1))).contiguous()
        hr_img = hr_img.to(dtype=torch.float32).div(255)
        
        return lr_img, hr_img

class LoadFastDataset(Dataset):
    def __init__(self, hr_images, scaling_factor, transform=None):
        self.hr_images = hr_images
        self.scaling_factor = scaling_factor
        self.transform = transform

    def __len__(self):
        return len(self.hr_images)

    def center_crop(self, hr_img, scaling_factor):
        height, width = hr_img.shape[0:2]

        # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
        x_remainder = width % (
            2 * scaling_factor if scaling_factor == 1.5 else scaling_factor
        )
        y_remainder = height % (
            2 * scaling_factor if scaling_factor == 1.5 else scaling_factor
        )
        left = int(x_remainder // 2)
        top = int(y_remainder // 2)
        right = int(left + (width - x_remainder))
        bottom = int(top + (height - y_remainder))
        hr_img = hr_img[top:bottom, left:right]
        return hr_img

    def __getitem__(self, idx):
        hr_img_path = self.hr_images[idx]
        
        if bool(self.transform):
            # hr_img = Image.open(hr_img_path).convert('RGB')
            hr_img = pickle.load(open(hr_img_path, 'rb'))
            hr_img = np.array(hr_img, dtype="float64")
            hr_img = self.transform(image=hr_img)
            hr_img = hr_img['image']
        else :
            hr_img = pickle.load(open(hr_img_path, 'rb'))
            hr_img = np.array(hr_img, dtype="float64")
            # hr_img = cv2.imread(hr_img_path)
            # hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            hr_img = self.center_crop(hr_img, self.scaling_factor)
            hr_img = np.array(hr_img, dtype="float64")

        # lr_img = hr_img.resize((int(hr_img.width // self.scaling_factor), int(hr_img.height // self.scaling_factor)), Image.BICUBIC)
        # return transforms.ToTensor()(lr_img), transforms.ToTensor()(hr_img)
        
        # hr_img = np.array(hr_img, dtype="float64")
        hr_height, hr_width = hr_img.shape[0:2]
        lr_img = imresize(hr_img, 1.0 / self.scaling_factor)  # equivalent to matlab's imresize
        lr_img = np.uint8(
            np.clip(lr_img, 0.0, 255.0)
        )  # this is to simulate matlab's imwrite operation
        hr_img = np.uint8(hr_img)

        lr_height, lr_width = lr_img.shape[0:2]

        # Sanity check
        assert (
            hr_width == lr_width * self.scaling_factor
            and hr_height == lr_height * self.scaling_factor
        )

        lr_img = torch.from_numpy(lr_img.transpose((2, 0, 1))).contiguous()
        lr_img = lr_img.to(dtype=torch.float32).div(255)

        hr_img = torch.from_numpy(hr_img.transpose((2, 0, 1))).contiguous()
        hr_img = hr_img.to(dtype=torch.float32).div(255)
        
        return lr_img, hr_img

class LoadValDataset(Dataset):
    def __init__(self, hr_images, scaling_factor, transform=None):
        self.hr_images = hr_images
        self.scaling_factor = scaling_factor
        self.transform = transform

    def __len__(self):
        return len(self.hr_images)

    def center_crop(self, hr_img, scaling_factor):
        height, width = hr_img.shape[0:2]

        # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
        x_remainder = width % (
            2 * scaling_factor if scaling_factor == 1.5 else scaling_factor
        )
        y_remainder = height % (
            2 * scaling_factor if scaling_factor == 1.5 else scaling_factor
        )
        left = int(x_remainder // 2)
        top = int(y_remainder // 2)
        right = int(left + (width - x_remainder))
        bottom = int(top + (height - y_remainder))
        hr_img = hr_img[top:bottom, left:right]
        return hr_img

    def __getitem__(self, idx):
        hr_img_path = self.hr_images[idx]
        fn_name = os.path.basename(hr_img_path.replace(" ", ""))
        img_name, _ = os.path.splitext(fn_name)
        if bool(self.transform):
            hr_img = Image.open(hr_img_path).convert('RGB')
            hr_img = np.array(hr_img, dtype="float64")
            hr_img = self.transform(image=hr_img)
            hr_img = hr_img['image']
        else : 
            hr_img = cv2.imread(hr_img_path)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            hr_img = self.center_crop(hr_img, self.scaling_factor)
            hr_img = np.array(hr_img, dtype="float64")
        
        hr_height, hr_width = hr_img.shape[0:2]
        lr_img = imresize(hr_img, 1.0 / self.scaling_factor)  # equivalent to matlab's imresize
        lr_img = np.uint8(
            np.clip(lr_img, 0.0, 255.0)
        )  # this is to simulate matlab's imwrite operation
        hr_img = np.uint8(hr_img)

        lr_height, lr_width = lr_img.shape[0:2]

        # Sanity check
        assert (
            hr_width == lr_width * self.scaling_factor
            and hr_height == lr_height * self.scaling_factor
        )

        lr_img = torch.from_numpy(lr_img.transpose((2, 0, 1))).contiguous()
        lr_img = lr_img.to(dtype=torch.float32).div(255)

        hr_img = torch.from_numpy(hr_img.transpose((2, 0, 1))).contiguous()
        hr_img = hr_img.to(dtype=torch.float32).div(255)
        
        return img_name, lr_img, hr_img
    
class LoadFastValDataset(Dataset):
    def __init__(self, hr_images, scaling_factor, transform=None):
        self.hr_images = hr_images
        self.scaling_factor = scaling_factor
        self.transform = transform

    def __len__(self):
        return len(self.hr_images)

    def center_crop(self, hr_img, scaling_factor):
        height, width = hr_img.shape[0:2]

        # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
        x_remainder = width % (
            2 * scaling_factor if scaling_factor == 1.5 else scaling_factor
        )
        y_remainder = height % (
            2 * scaling_factor if scaling_factor == 1.5 else scaling_factor
        )
        left = int(x_remainder // 2)
        top = int(y_remainder // 2)
        right = int(left + (width - x_remainder))
        bottom = int(top + (height - y_remainder))
        hr_img = hr_img[top:bottom, left:right]
        return hr_img

    def __getitem__(self, idx):
        hr_img_path = self.hr_images[idx]
        fn_name = os.path.basename(hr_img_path.replace(" ", ""))
        img_name, _ = os.path.splitext(fn_name)        
        if bool(self.transform):
            hr_img = pickle.load(open(hr_img_path, 'rb'))
            hr_img = np.array(hr_img, dtype="float64")
            hr_img = self.transform(image=hr_img)
            hr_img = hr_img['image']
        else :
            hr_img = pickle.load(open(hr_img_path, 'rb'))
            hr_img = np.array(hr_img, dtype="float64")
            hr_img = self.center_crop(hr_img, self.scaling_factor)
            hr_img = np.array(hr_img, dtype="float64")
        
        hr_height, hr_width = hr_img.shape[0:2]
        lr_img = imresize(hr_img, 1.0 / self.scaling_factor)  # equivalent to matlab's imresize
        lr_img = np.uint8(
            np.clip(lr_img, 0.0, 255.0)
        )  # this is to simulate matlab's imwrite operation
        hr_img = np.uint8(hr_img)

        lr_height, lr_width = lr_img.shape[0:2]

        # Sanity check
        assert (
            hr_width == lr_width * self.scaling_factor
            and hr_height == lr_height * self.scaling_factor
        )

        lr_img = torch.from_numpy(lr_img.transpose((2, 0, 1))).contiguous()
        lr_img = lr_img.to(dtype=torch.float32).div(255)

        hr_img = torch.from_numpy(hr_img.transpose((2, 0, 1))).contiguous()
        hr_img = hr_img.to(dtype=torch.float32).div(255)
        
        return img_name, lr_img, hr_img
