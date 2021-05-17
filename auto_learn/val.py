import torch.nn as nn
import torch
import cv2
import numpy as np
import json
from pycocotools.coco import COCO
import os
from copy import deepcopy
from public.detection.models import yolof2
from public.detection.models.decode import FCOSDecoder
from tqdm import tqdm
import random
import sys
import time

use_gpu = True
model_dir = "/home/jovyan/data-vol-polefs-1/small_sample/checkpoints/no_freeze_monkey/v1/best.pth"
im_dir = '/home/jovyan/data-vol-polefs-1/small_sample/dataset/monkey/input_monkey/test'

#加载模型 
torch.cuda.set_device(2)

model = yolof2.resnet50_yolof(num_classes=1)
pre_model = torch.load(model_dir, map_location="cpu")
model.load_state_dict(pre_model, strict=False)
model.eval()


resize = 667
im_list = os.listdir(im_dir)[1:]
decoder = FCOSDecoder(image_w=resize, image_h=resize,strides=[16])

out = {"scores":[], "bbox":[]}
if use_gpu:
    decoder = decoder.cuda()
    model = model.cuda()
    

with torch.no_grad():
    for item in tqdm(im_list):
        current_dir = os.path.join(im_dir, item)
        img = cv2.imdecode(np.fromfile(current_dir, dtype=np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        height, width, _ = img.shape
        max_image_size = max(height, width)
        resize_factor = resize / max_image_size
        resize_height, resize_width = int(height * resize_factor), int(
            width * resize_factor)
        img = cv2.resize(img, (resize_width, resize_height))
        resized_img = np.zeros((resize, resize, 3))
        resized_img[0:resize_height, 0:resize_width] = img
        resized_img = torch.tensor(resized_img).permute(2, 0, 1).float().unsqueeze(0)
        resized_img = resized_img.cuda()
        cls_heads, reg_heads, center_heads, batch_positions = model(resized_img)
        scores, classes, boxes = decoder(cls_heads, reg_heads, center_heads, batch_positions)
        scores, classes, boxes = scores.cpu(), classes.cpu(), boxes.cpu()
        score = scores[0][0].item()
        bbox = (boxes[0][0]/resize_factor).tolist()
        if score > 0.2:
            out["scores"].append(score)
            out["bbox"].append(bbox)
        draw = cv2.imdecode(np.fromfile(current_dir, dtype=np.uint8), -1)
        cv2.rectangle(draw, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (225, 0, 0), 2)
        cv2.imwrite(im_dir+"/out_"+item,draw)

json.dump(out, open("out.json", "w"))
