from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listKITTIfiles as listLoader
from dataloader import KITTILoader as dataLoader
import cv2
from models import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description='Guided-Stereo')
parser.add_argument('--datapath', default='2011_09_26_0011/',
                    help='datapath')
parser.add_argument('--loadmodel', default='weights/psmnet-ft.tar',
                    help='load model')
parser.add_argument('--output_dir', default='results/',
                    help='output directory')
parser.add_argument('--guided', action='store_true', default=False, help='Enable guided stereo')
parser.add_argument('--display', action='store_true', default=False, help='Display output')
parser.add_argument('--save', action='store_true', default=False, help='Save output')
parser.add_argument('--verbose', action='store_true', default=False, help='Print stats for each single image')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

all_left, all_right, all_guide, all_disp = listLoader.dataloader(args.datapath)

TestImgLoader = torch.utils.data.DataLoader(
         dataLoader.imageLoader(all_left, all_right, all_guide, all_disp), 
         batch_size= 1, shuffle= False, num_workers= 4, drop_last=False)

# build model
model = psmnet(192, args.guided)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

# Running guided stereo!
def test(reference,imgL,imgR,guideL,disp_true,h,w,batch_idx):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   

        validhints = (guideL > 0).float()

        #computing density
        density=(float(torch.nonzero(validhints).size(0)) / ((validhints.size(1))*validhints.size(2))*100.)

        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            output3 = model(imgL,imgR,guideL,validhints,k=10,c=1)

        top_pad   = 384-h
        left_pad  = 1280-w

        pred_disp = output3.data.cpu()
        if args.display or args.save:
            display_and_save(batch_idx, reference*255, guideL, pred_disp, disp_true, top_pad, left_pad)

        true_disp_nog = disp_true.clone()
        true_disp_all = disp_true.clone()

        # compute NoG error
        index_nog = np.argwhere(true_disp_nog*(1-validhints)>0)
        true_disp_nog[index_nog[0][:], index_nog[1][:], index_nog[2][:]] = np.abs(true_disp_nog[index_nog[0][:], index_nog[1][:], index_nog[2][:]]-pred_disp[index_nog[0][:], index_nog[1][:], index_nog[2][:]])
        correct2_nog = (true_disp_nog[index_nog[0][:], index_nog[1][:], index_nog[2][:]] < 2)
        bad2_nog = 1-(float(torch.sum(correct2_nog))/float(len(index_nog[0])))
        avg_nog  = float(torch.sum(true_disp_nog[index_nog[0][:], index_nog[1][:], index_nog[2][:]])/float(len(index_nog[0])))

        # compute All error
        index_all = np.argwhere(true_disp_all>0)
        true_disp_all[index_all[0][:], index_all[1][:], index_all[2][:]] = np.abs(true_disp_all[index_all[0][:], index_all[1][:], index_all[2][:]]-pred_disp[index_all[0][:], index_all[1][:], index_all[2][:]])
        correct2_all = (true_disp_all[index_all[0][:], index_all[1][:], index_all[2][:]] < 2)
        bad2_all = 1-(float(torch.sum(correct2_all))/float(len(index_all[0])))
        avg_all  = float(torch.sum(true_disp_all[index_all[0][:], index_all[1][:], index_all[2][:]])/float(len(index_all[0])))

        return bad2_all, bad2_nog, avg_all, avg_nog, density

# Dirty work to show/save results...
def display_and_save(batch_idx, left, guide, disparity, gt, top_pad, left_pad):
        left_2show = np.transpose(left.cpu().numpy()[0][:,top_pad:,left_pad:], (1,2,0)).astype(np.uint8)
        left_2show = cv2.cvtColor(left_2show, cv2.COLOR_BGR2RGB)
        disp_2show = cv2.applyColorMap(np.clip(50+2*disparity.numpy()[0][top_pad:,left_pad:], a_min=0, a_max=255.).astype(np.uint8), cv2.COLORMAP_JET)
        guide_2show = cv2.applyColorMap(np.clip(50+2*guide.numpy()[0][top_pad:,left_pad:], a_min=0, a_max=255.).astype(np.uint8), cv2.COLORMAP_JET) * np.expand_dims((guide.numpy()[0][top_pad:,left_pad:]>0),-1)
        gt_2show = cv2.applyColorMap(np.clip(50+2*gt.numpy()[0][top_pad:,left_pad:], a_min=0, a_max=255.).astype(np.uint8), cv2.COLORMAP_JET)* np.expand_dims((gt.numpy()[0][top_pad:,left_pad:]>0),-1)

        collage = np.concatenate((np.concatenate((left_2show, guide_2show), 1), np.concatenate((disp_2show, gt_2show), 1)), 0)
        collage = cv2.resize(collage, (collage.shape[1]//2, collage.shape[0]//2))
        if args.display:
            cv2.imshow("Guided stereo", collage)
            k = cv2.waitKey(10)
            if k == 27: 
                sys.exit()  # esc to quit
        if args.save:
            cv2.imwrite(args.output_dir+"/%06d.png"%batch_idx, collage)

# main
def main():

	total_bad2_all = 0
	total_bad2_nog = 0
        total_avg_all = 0
        total_avg_nog = 0
        total_density = 0

        if not os.path.exists(args.output_dir) and args.save:
           os.mkdir(args.output_dir)

        ## Test ##
        if args.verbose:
            print('Frame & bad2-all & bad2-NoG & MAE-all & MAE-NoG & density')
        for batch_idx, (reference, imgL, imgR, guideL, dispL, h, w) in enumerate(TestImgLoader):
            bad2_all, bad2_nog, avg_all, avg_nog, density = test(reference,imgL,imgR,guideL,dispL,h,w,batch_idx)
            total_bad2_all += bad2_all
            total_bad2_nog += bad2_nog
            total_avg_all  += avg_all
            total_avg_nog  += avg_nog
            total_density += density
            if args.verbose:
                print('%06d & %.2f & %.2f & %.2f & %.2f & %.2f' %(batch_idx, bad2_all*100., bad2_nog*100., avg_all, avg_nog, density))

        print("bad2-all & bad2-NoG & MAE-all & MAE-NoG & density")
        print('%.2f & %.2f & %.2f & %.2f & %.2f' % ((total_bad2_all/len(TestImgLoader)*100), (total_bad2_nog/len(TestImgLoader)*100), (total_avg_all/len(TestImgLoader)), (total_avg_nog/len(TestImgLoader)), (total_density/len(TestImgLoader))))

if __name__ == '__main__':
   main()
