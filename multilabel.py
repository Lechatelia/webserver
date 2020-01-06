# this is for the DNN model of multilabl classfication
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import sys
import copy
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import scipy
import scipy.io
import pdb
import argparse
import matplotlib.pyplot as plt
import cv2
import random

classes_list = ['aeroplane', 'bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
def parse_args():
    parser = argparse.ArgumentParser(description='Train a multi label')
    parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=2, type=int)
    parser.add_argument('--checkpoint', default="../multi-label/VOC2007/model/model_ft.pt", type=str, help='pretrained model name')
    parser.add_argument('--save_images', default=True, type=bool, help='whether to save predicted image')
    parser.add_argument('--threshold', default=0.5, type=float, help='threshod of predicted value')
    parser.add_argument('--gpu_number', default=1, type=int, help='which one gpu to use')
    parser.add_argument('--output_dir', default='./output', type=str, help='where to save predictions')
    # parser.add_argument('--predict_number', default=20, type=int, help='how many images do you want to test')
    args = parser.parse_args()
    return args


class Multilabel():
    ''' multilabel classfication'''
    def __init__(self, gpu_number=None,img_size=448):
        self.args = parse_args()
        self.img_size = img_size
        if gpu_number is not None:
            self.args.gpu_number = gpu_number
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        self.device = torch.device("cuda:{}".format(self.args.gpu_number) if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = torch.load(self.args.checkpoint)
        self.model = self.model.to(self.device)
        self.list2class = classes_list

        self.transforms =transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.RandomResizedCrop((224)),
        transforms.Resize((self.img_size,self.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def detect(self, image):
        if not image.mode=='RGB':
            image.convert('RGB')
        image = self.transforms(image).unsqueeze(0)

        image = image.to(self.device).float()
        self.model.eval()
        # forward
        with torch.set_grad_enabled(False):
            try:
                outputs = self.model(image)
            except:
                return {}

            outputs = torch.sigmoid(outputs)
            for i in range(1):
                output = outputs[i]
                indices = torch.nonzero(output.gt(self.args.threshold))
                indices = indices.cpu().numpy().squeeze(1)
                results = [{'name':self.list2class[l], 'confidence':int(output[l].item()*100) } for l in indices]
            return  results


    def test(self, path):
        image = Image.open(path)

        results = self.detect(image)
        return results



if __name__ == "__main__":
    detector = Multilabel()
    results = detector.test('./images/img1.jpg')
    print(results)

