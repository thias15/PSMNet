from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from dataloader import preprocess 
from models import *

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--dataset', default='KITTI15',
                    help='KITTI version')
parser.add_argument('--datapath', default='C:/Github/PSMNet/data_scene_flow/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maximum disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if 'KITTI' in args.dataset:
   from dataloader import KITTI_submission_loader as DA
elif args.dataset == 'AirSim':
   from dataloader import AirSimFiles as DA
elif args.dataset == 'SceneFlow':
   from dataloader import SceneFlowFiles as DA

if 'KITTI' in args.dataset:
    test_left_img, test_right_img = DA.dataloader(args.datapath)
else:
    test_left_img, test_right_img, _, _, _, _ = DA.dataloader(args.datapath)
    #_, _, _, test_left_img, test_right_img, _ = DA.dataloader(args.datapath)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

        imgL, imgR
        with torch.no_grad():
            output = model(imgL,imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()

        return pred_disp


def main():
   processed = preprocess.get_transform(augment=False)
   num_imgs = len(test_left_img)
   for inx in range(num_imgs):

       imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
       imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))
       imgL = processed(imgL_o).numpy()
       imgR = processed(imgR_o).numpy()
       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

       if 'KITTI' in args.dataset:
           # pad to (384, 1248)
           top_pad = 384-imgL.shape[2]
           left_pad = 1248-imgL.shape[3]
           imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
           imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       else:
           # pad to (576, 960)
           top_pad = 576-imgL.shape[2]
           left_pad = 960-imgL.shape[3]
           imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
           imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

       start_time = time.time()
       pred_disp = test(imgL,imgR)
       print('processed %d/%d, time = %.2f' %(inx, num_imgs, time.time() - start_time))

       if 'KITTI' in args.dataset:
           # crop to original size
           top_pad   = 384-imgL_o.shape[0]
           left_pad  = 1248-imgL_o.shape[1]
           img = pred_disp[top_pad:,:-left_pad]
       else:
           # crop to original size
           top_pad   = 576-imgL_o.shape[0]
           #left_pad = 960 - imgL_o.shape[1]
           img = pred_disp[top_pad:,:]
           #img = pred_disp
          

       path = os.path.join(*test_left_img[inx].replace("\\", "/").split('/')[:-2], 'disparity_psm')
       file = test_left_img[inx].replace("\\", "/").split('/')[-1]

       if not os.path.exists(path):
           os.makedirs(path)
       skimage.io.imsave(os.path.join(path,file), (img * 256).astype('uint16'))

       #skimage.io.imsave(file, (img * 256).astype('uint16'))


if __name__ == '__main__':
   main()






