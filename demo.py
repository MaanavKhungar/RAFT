import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import torchvision

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    # print(img_flo.shape)
    # print(type(img_flo))
    img_flo2 = tf.expand_dims(img_flo, axis=0) 
    # print(img_flo2.shape)
    # print(type(img_flo2))
    
    # plt.imshow(img_flo / 255.0)
    # plt.show()


    logdir = "logs/train_data/" + datetime.now().strftime("%d%b%Y-%I%M%p")
    
    # tensorboard_logger = SummaryWriter(logdir)
    # tensorboard_logger.add_image('Segmentation (Ground Truth)', 
    #                      img_flo / 255.0)

    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    # Using the file writer, log the reshaped image.
    with file_writer.as_default():
        tf.summary.image("Training data", img_flo2 / 255.0, step=0)

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
