from numpy import random
import torch
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import os
import shutil

# Initialize
out = r'inference\output'
set_logging()
device = select_device('')
if os.path.exists(out):
    shutil.rmtree(out)  # delete output folder
os.makedirs(out)  # make new output folder
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load('weights/yolov5s.pt', map_location=device)  # load FP32 model
imgsz = check_img_size(512, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16

# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
    modelc.to(device).eval()

# Set Dataloader


# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once


def PoseDect(path, imgsz=512):
    res = []
    dataset = LoadImages(path, img_size=imgsz)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    x, y, w, h = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    res.append([names[int(cls)], float(conf), x, y, w, h])

    return res

    # for _, img, im0s, _ in dataset:
    #
    #     img = torch.from_numpy(img).to(device)
    #     img = img.half() if half else img.float()  # uint8 to fp16/32
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)
    #
    #     pred = model(img, augment=False)[0]
    #     # Apply NMS
    #     pred = non_max_suppression(pred, 0.4, .05, classes=None, agnostic=None)
    #
    #     for i, det in enumerate(pred):  # detections per image
    #         p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
    #
    #         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #         if det is not None and len(det):
    #             # Rescale boxes from img_size to im0 size
    #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    #
    #             # Write results
    #             for *xyxy, conf, cls in reversed(det):
    #                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    #                 x, y, w, h = xywh
    #                 if(int(cls)==0):
    #                     res.append([names[int(cls)], float(conf), x, y, w, h])
    #                 #res.append([int(cls), float(conf), x, y, w, h])
    #
    #                 # draw
    #                 # label = '%s %.2f' % (names[int(cls)], conf)
    #                 #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
    #
    # return res


if __name__ == '__main__':

    path = r'inference\images\0152498D-225A-4126-AEBE-B6D9423E12E7.png'
    s = PoseDect(path=path)
    print(s)
    import cv2

    img = cv2.imread(r'inference\images\0152498D-225A-4126-AEBE-B6D9423E12E7.png')
    for box in s:
        x1, y1, x2, y2 = box[2:]
        # 映射原图尺寸
        x = int(x1 * img.shape[1])
        y = int(y1 * img.shape[0])
        w = int(x2 * img.shape[1])
        h = int(y2 * img.shape[0])
        # 计算出左上角和右下角：原x,y是矩形框的中心点
        a = int(x - w / 2)
        b = int(y - h / 2)
        c = int(x + w / 2)
        d = int(y + h / 2)

        print(x1, y1, x1 + x2, y1 + y2)
        print(x, y, x + w, y + h)
        print(a, b, c, d)

        cv2.rectangle(img, (a, b), (c, d), (255, 0, 0), 2)
    cv2.imshow('dst', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
