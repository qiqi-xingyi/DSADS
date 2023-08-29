import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import os
import shutil

################################################################################################
def PoseDect(im0s , model ,imgsz=640):

    set_logging()
    device = select_device('0')

    half = device.type != 'cpu'

    # Load model
    #model = attempt_load('weights/yolov5x.pt', map_location=device)
    imgsz = check_img_size(640, s=model.stride.max())
    if half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])
        model.to(device).eval()

    names = model.module.names if hasattr(model, 'module') else model.names

    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    # print(img)

    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

# #################################################################################################

    # dataset = LoadImages(path , img_size=imgsz)
    # for path, img, im0s, vid_cap in dataset:

    #img = ?
    img = letterbox(im0s, new_shape=640)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    ########################################### Sizes of tensors must match except in dimension 3
    pred = model(img, augment=False)[0]  #######
    # print("PPPPPPP", pred)
    ###########################################
    # print(img.ndimension())
    # print(img)

    # Apply NMS
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
    # t2 = time_synchronized()
    #print(pred)
    # Apply Classifier
    #classify = False

    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)


    # Process detections
    for i, det in enumerate(pred):  # detections per image
        #print("2")
        gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #print(det)
        box_points = []
        all_points = []  # del
        if det is not None and len(det):
            #print("3")
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            mynames = ['person', 'car', 'truck', 'bicyle', 'bus', 'train', 'motorcycle', 'traffic light',
                       'fire hydrant', 'stop sign',
                       'parking meter', 'potted plant', 'dining table']


            for *xyxy, conf, cls in reversed(det):
                x,y,w,h = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                label = f'{names[int(cls)]} {conf:.2f}'
                #label = f'{names[int(cls)]} {conf:.2f}'

                # x1, y1, x2, y2 = xyxy
                # x1, y1, x2, y2 = x1.int(), y1.int(), x2.int(), y2.int()
                # print("x1, y1, x2, y2", x1, y1, x2, y2)
                #x1, y1, x2, y2 = int(x[0]), int(x[1]), int(x[2], int(x[3])

                if (names[int(cls)] in mynames):
                    # top = y
                    # left = x
                    # bottom = y + h
                    # right = x + w
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    left = c1[0]
                    top = c1[1]
                    right = c2[0]
                    bottom = c2[1]

                    all_points.append([left, top, right, bottom, names[int(cls)]])
                    box_points.append([(left + right) / 2, bottom])
                    colors = None
                    if names[int(cls)] == 'person':
                        colors = (24, 134, 53)
                    elif names[int(cls)] == 'car':
                        colors = (189, 52, 46)
                    elif names[int(cls)] == 'motorcycle':
                        colors = (1, 54, 184)

                    plot_one_box(xyxy, im0s, label=label, color=colors, line_thickness=3)

                #print(names[int(cls)] in mynames, names[int(cls)])


    return im0s, box_points, all_points
