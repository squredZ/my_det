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
    #**********************************************
    #输入verison
    version = 1
    #**********************************************
    
    log = './log_' + str(version)  # Path to save log
    checkpoint_path = '/home/jovyan/data-vol-polefs-1/small_sample/checkpoints/v{}'.format(version)  # Path to store checkpoint model
    resume = '/home/jovyan/data-vol-polefs-1/small_sample/checkpoints/v{}/latest.pth'.format(version)  # load checkpoint model
    evaluate = None  # evaluate model path
    base_path = '/home/jovyan/data-vol-polefs-1/small_sample/dataset'
    train_dataset_path = os.path.join(base_path, 'images/images')
    val_dataset_path = os.path.join(base_path, 'images/images')
    dataset_annotations_path = os.path.join(base_path, 'annotations/current')

    network = "resnet50_yolof"
    pretrained = True
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
    strides = [16]
    scales = [1.0]


    epochs = 48
    per_node_batch_size = 2
    lr = 1e-4
    num_workers = 4
    print_interval = 1
    apex = True
    sync_bn = False