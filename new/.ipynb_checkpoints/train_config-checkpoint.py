import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append("../")

from public.detection.dataset.cocodataset import CocoDetection, Resize, RandomFlip, RandomCrop, RandomTranslate, Normalize

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config(object):
    log = './log'  # Path to save log
    checkpoint_path = '/home/jovyan/data-vol-polefs-1/yolof_dc5_coco_1333'  # Path to store checkpoint model
    resume = '/home/jovyan/data-vol-polefs-1/yolof_dc5_coco_1333/latest.pth'  # load checkpoint model
    evaluate = None  # evaluate model path
    base_path = '/home/jovyan/data-vol-polefs-1/dataset/coco'
    train_dataset_path = os.path.join(base_path, 'train2017')
    val_dataset_path = os.path.join(base_path, 'val2017')
    dataset_annotations_path = os.path.join(base_path, 'annotations')
    
    
    
    network = "resnet50_yolofdc5"
    seed = 0
    input_image_size = 1333

    train_dataset = CocoDetection(image_root_dir=train_dataset_path,
                                  annotation_root_dir=dataset_annotations_path,
                                  set="train2017",
                                  transform=transforms.Compose([
                                      RandomFlip(flip_prob=0.5),
                                      RandomCrop(crop_prob=0.5),
                                      RandomTranslate(translate_prob=0.5),
                                      Normalize(mean=[0.40789654, 0.44719302, 0.47026115],std=[0.28863828, 0.27408164, 0.27809835]),
                                      Resize(resize=input_image_size)
                                  ]))
    val_dataset = CocoDetection(image_root_dir=val_dataset_path,
                                annotation_root_dir=dataset_annotations_path,
                                set="val2017",
                                transform=transforms.Compose([
                                    Normalize(),
                                    Resize(resize=input_image_size),
                                ]))
    epochs = 24
    per_node_batch_size = 4
    lr = 1e-4
    num_workers = 4
    print_interval = 100
    apex = True
    sync_bn = False

    pretrained=True
    freeze_stage_1=True
    freeze_bn=True




    #fpn
    use_yolof = True
    C5_inplanes = 512 * 4
    fpn_out_channels=512
    use_p5=True
    
    #head
    class_num=80
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True
    head_planes = 512

    #training
    strides=[16]
    limit_range=[[-1,512]]
    scales = [1.0]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000