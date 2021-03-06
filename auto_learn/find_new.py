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

version = int(sys.argv[1])
save_dir = sys.argv[2]
model_load_dir = sys.argv[3]
use_gpu = True

#model path
model_dir = model_load_dir + "/v{}/best.pth".format(version-1)

#images to find
im_dir = "/home/jovyan/data-vol-polefs-1/small_sample/dataset/images/images/"

#the old training annotation path
old_train_dir = os.path.join(save_dir, "instances_train.json")
#the sample pool
wait_dir = "/home/jovyan/data-vol-polefs-1/small_sample/dataset/annotations/instances_wait.json"

#加载模型 
torch.cuda.set_device(2)

model = yolof2.resnet50_yolof(num_classes=1)
pre_model = torch.load(model_dir, map_location="cpu")
model.load_state_dict(pre_model, strict=False)
model.eval()


#加载json
with open(wait_dir, "r") as f:
    wait_json = json.load(f)
wait_coco = COCO(wait_dir)


with open(old_train_dir, "r") as f:
    old_train = json.load(f)
old_file = [item["file_name"] for item in old_train["images"]]
new_train = deepcopy(old_train)


resize = 667
im_list = [[item["file_name"],item["id"]] for item in wait_json["images"]]
decoder = FCOSDecoder(image_w=resize, image_h=resize,strides=[16])
im_scores = {}


if use_gpu:
    decoder = decoder.cuda()
    model = model.cuda()
    

with torch.no_grad():
    for item in tqdm(im_list):
        current_dir = os.path.join(im_dir, item[0])
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
        score = scores[0].mean().item()

        if score > 0.9 or score < 0.1:
            continue
        entropy = -(score*np.log(score) + (1-score)*np.log(1-score))
        im_scores[item[1]] = entropy

        
        
def sort_score(x):
    try:
        out = im_scores[x[1]]
    except:
        out = -9999
    return out
im_list.sort(key=sort_score, reverse=True)
# random.shuffle(im_list)

im_ids = []

count = 0
idx = 0
while count < 200:
    item = im_list[idx]
    idx += 1
    if item[0] in old_file:
        continue
    else:
        im_ids.append(item[1])
        count += 1
    



new_train["images"] = new_train["images"] + wait_coco.loadImgs(ids=im_ids)
new_train["annotations"] = new_train["annotations"] + wait_coco.loadAnns(wait_coco.getAnnIds(imgIds=im_ids))
# new_train["images"] = wait_coco.loadImgs(ids=im_ids)
# new_train["annotations"] = wait_coco.loadAnns(wait_coco.getAnnIds(imgIds=im_ids))

json.dump(new_train, open(os.path.join(save_dir, "instances_train.json"), 'w'))
json.dump(new_train, open(os.path.join(save_dir, "instances_train_v{}.json".format(version)), 'w'))
