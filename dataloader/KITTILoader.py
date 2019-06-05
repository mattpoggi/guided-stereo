import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
import preprocess 

class imageLoader(data.Dataset):
    def __init__(self, left, right, guide, disparity):
 
        self.left = left
        self.right = right
        self.disp_L = disparity
        self.guide = guide

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = Image.open(left).convert('RGB')
        right_img = Image.open(right).convert('RGB')
        dataL = Image.open(disp_L)
        guideL = Image.open(self.guide[index])

        w, h = left_img.size

        left_img = left_img.crop((w-1280, h-384, w, h))
        right_img = right_img.crop((w-1280, h-384, w, h))
        w1, h1 = left_img.size

        dataL = dataL.crop((w-1280, h-384, w, h))
        dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

        guideL = guideL.crop((w-1280, h-384, w, h))
        guideL = np.ascontiguousarray(guideL,dtype=np.float32)/256

        processed = preprocess.get_transform(augment=False)  
        rawimage = preprocess.identity(256)

        reference = rawimage(left_img)
        left_img       = processed(left_img)
        right_img      = processed(right_img)
          
        return reference, left_img, right_img, guideL, dataL, h, w

    def __len__(self):
        return len(self.left)
