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
    log = './log_1'  # Path to save log
    checkpoint_path = '/home/jovyan/data-vol-polefs-1/small_sample/checkpoints/v1'  # Path to store checkpoint model
    resume = '/home/jovyan/data-vol-polefs-1/small_sample/checkpoints/v1/latest.pth'  # load checkpoint model
    evaluate = None  # evaluate model path
    base_path = '/home/jovyan/data-vol-polefs-1/small_sample/dataset'
    train_dataset_path = os.path.join(base_path, 'images/xtrzk')
    val_dataset_path = os.path.join(base_path, 'images/xtrzk')
    dataset_annotations_path = os.path.join(base_path, 'annotations/v1')

    network = "resnet50_fcos"
    pretrained = False
    num_classes = 1
    seed = 0
    input_image_size = 667

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
    fpn_bn = False
    use_gn = False
    use_TransConv = False
    use_YolofDC5 = True
    freeze = True


    epochs = 12
    per_node_batch_size = 12
    lr = 1e-5
    num_workers = 4
    print_interval = 100
    apex = True
    sync_bn = False