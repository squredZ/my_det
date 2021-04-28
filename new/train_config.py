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
    checkpoint_path = '/home/jovyan/data-vol-polefs-1/yolof2.1_res18_crops_checkpoints_dataV4_boso0.3_withImPre'  # Path to store checkpoint model
    resume = '/home/jovyan/data-vol-polefs-1/yolof2.1_res18_crops_checkpoints_dataV4_boso0.3_withImPre/latest.pth'  # load checkpoint model
    evaluate = None  # evaluate model path
    base_path = '/home/jovyan/data-vol-polefs-1/crop_dataset'
    train_dataset_path = os.path.join(base_path, 'images_v4_boso_30')
    val_dataset_path = os.path.join('/home/jovyan/data-vol-polefs-1/dataset', 'images/images')
    dataset_annotations_path = os.path.join(base_path, 'annotations')

    seed = 0
    input_image_size = 1333

    train_dataset = CocoDetection(image_root_dir=train_dataset_path,
                                  annotation_root_dir=dataset_annotations_path,
                                  set="train",
                                  transform=transforms.Compose([
                                      RandomFlip(flip_prob=0.5),
                                      RandomCrop(crop_prob=0.5),
                                      RandomTranslate(translate_prob=0.5),
                                      Normalize(),
                                      Resize(resize=input_image_size),
                                  ]))
    val_dataset = CocoDetection(image_root_dir=val_dataset_path,
                                annotation_root_dir=dataset_annotations_path,
                                set="val",
                                transform=transforms.Compose([
                                    Normalize(),
                                    Resize(resize=input_image_size),
                                ]))
    epochs = 24
    per_node_batch_size = 2
    lr = 1e-2
    num_workers = 4
    print_interval = 100
    apex = True
    sync_bn = False

    pretrained=False
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

    #training
    strides=[16]
    limit_range=[[-1,512]]
    scales = [1.0]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000