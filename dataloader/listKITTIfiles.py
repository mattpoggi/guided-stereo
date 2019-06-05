import torch.utils.data as data
import numpy as np
import os

def dataloader(filepath):

  left_fold  = 'image_2/'
  right_fold = 'image_3/'
  disp_L = 'disp_occ_0/'
  guide_L = 'lidar/'

  images = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
  images.sort()

  left  = [filepath+left_fold+img for img in images]
  right = [filepath+right_fold+img for img in images]
  disp = [filepath+disp_L+img for img in images]
  guide =  [filepath+guide_L+img for img in images]
  
  return left, right, guide, disp
