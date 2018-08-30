import torch.utils.data as data

from PIL import Image
import os
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  subsets = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]

  subsets = ['NH']
   
  left_folder  = 'left'
  right_folder = 'right'
  disp_folder = 'disparity_gt'
  left = []
  right = []
  disp = []

  for i in range(0,len(subsets)):
    for img in os.listdir(os.path.join(filepath, subsets[i], left_folder)):
        left.append(os.path.join(filepath, subsets[i], left_folder, img))
    for img in os.listdir(os.path.join(filepath, subsets[i], right_folder)):
        right.append(os.path.join(filepath, subsets[i], right_folder, img)) 
    for img in os.listdir(os.path.join(filepath, subsets[i], disp_folder)):
        disp.append(os.path.join(filepath, subsets[i], disp_folder, img))

  split_idx = int(len(left)*0.8)
  left_train  = left[:split_idx]
  left_val = left[split_idx:]
  right_train  = right[:split_idx]
  right_val = right[split_idx:]
  disp_train  = disp[:split_idx]
  disp_val = disp[split_idx:]

  return left_train, right_train, disp_train, left_val, right_val, disp_val
