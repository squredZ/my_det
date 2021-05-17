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
    def __init__(self,ver=1):
        
        #**********************************************
        self.version = ver
        #**********************************************
        self.name = "monkey" #training name, to save log and check_points, set as your willing
        self.log = './log_' + self.name + '/log_' + str(self.version)  # Path to save log
        self.checkpoint_path = '/home/jovyan/data-vol-polefs-1/small_sample/checkpoints/{}/v{}'.format(self.name,self.version)  # Path to store checkpoint model
        self.resume = '/home/jovyan/data-vol-polefs-1/small_sample/checkpoints/{}/v{}/latest.pth'.format(self.name,self.version)  # load checkpoint model
        self.pre_model_dir = '/home/jovyan/data-vol-polefs-1/small_sample/checkpoints/{}/v{}'.format(self.name, self.version-1)
        self.evaluate = None  # evaluate model path
        self.base_path = '/home/jovyan/data-vol-polefs-1/small_sample/dataset/monkey/input_monkey'
        self.train_dataset_path = os.path.join(self.base_path, 'train2007')
        self.val_dataset_path = os.path.join(self.base_path, 'val2007')
        self.dataset_annotations_path = os.path.join(self.base_path, 'annotations')

        self.network = "resnet50_yolof"
        #use the pretrained model
        self.pretrained = True
        #freeze the backbone and neck
        self.freeze = False
        #use the previous model as pretrained model when auto_training
        self.previous = True
        #class num
        self.num_classes = 1
        self.seed = 0
        #resize shape
        self.input_image_size = 667

        self.train_dataset = CocoDetection(image_root_dir=self.train_dataset_path,
                                      annotation_root_dir=self.dataset_annotations_path,
                                      set="train",
                                      transform=transforms.Compose([
                                          RandomFlip(flip_prob=0.5),
                                          RandomCrop(crop_prob=0.5),
                                          RandomTranslate(translate_prob=0.5),
                                          Normalize(),
                                          Resize(resize=self.input_image_size),
                                      ]))
        self.val_dataset = CocoDetection(image_root_dir=self.val_dataset_path,
                                    annotation_root_dir=self.dataset_annotations_path,
                                    set="val",
                                    transform=transforms.Compose([
                                        Normalize(),
                                        Resize(resize=self.input_image_size),
                                    ]))
        self.fpn_bn = False
        self.use_gn = False
        self.use_TransConv = False
        self.use_YolofDC5 = True

        self.strides = [16]
        self.scales = [1.0]


        self.epochs = 48
        self.per_node_batch_size = 2
        self.lr = 1e-5
        self.num_workers = 4
        self.print_interval = 10
        self.apex = True
        self.sync_bn = False