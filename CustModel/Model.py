import torch
from utils.datasets import LoadImages, letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device,time_synchronized

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from utils.plots import plot_one_box,plot_one_box_PIL
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
import numpy as np
from pathlib import Path
import warnings
import os

class Yolov7:
    def __init__(self, weights, img_size=640, conf_thres=0.3, iou_thres=0.45, dev=''):
        warnings.filterwarnings("ignore")
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        self.device = select_device(dev)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(img_size, s=self.stride)  # check img_size
        trace = not "store_true"
        if trace:
            self.model = TracedModel(self.model, self.device, img_size)
        if self.half:
            self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1
        
    def predict(self, path, save_img=False):
        img0 = cv2.imread(path)
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or \
                                          self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=not "")[0]  

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment='store_true')[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        t3 = time_synchronized()
        labels,conf_ = [], []
        cordinates = []
        for i, det in enumerate(pred):
            s = ''
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                    label = f'{self.names[int(cls)]}'
                    plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=1)
                    labels.append(label)
                    cordinates.append(torch.tensor(xyxy).view(-1).tolist())
                    conf_.append(f'{conf:.2f}')
        if save_img:
            path_, filename = os.path.split(path)
            newpath = os.path.join(path_, f"boxes_{filename}")
            cv2.imwrite(newpath, img0)
        return labels, cordinates, conf_, s