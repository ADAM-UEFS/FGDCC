
import os
import subprocess
import time

import numpy as np

from logging import getLogger

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()


import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform

def build_dataset(is_train, image_folder):
    transform = build_transform(is_train)
    root = os.path.join(image_folder, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset

def build_transform(is_train):

    #mean = (0.436, 0.444, 0.330) # PlantCLEF2022 stats
    #std = (0.203, 0.199, 0.195)

    # Testing imagenet stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    input_size=224
    # train transform
    if is_train:
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.0,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1, 
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def make_GenericDataset(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    feature_extraction=False,
    subset_file=None
):
    
    index_targets = False 
    
    dataset = build_dataset(is_train=training, image_folder=image_folder)

    #print(dataset.class_to_idx) 

    logger.info('Finetuning dataset created')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)    
    return dataset, data_loader, dist_sampler

class PC2022(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        image_folder='/home/rtcalumby/adam/luciano/LifeCLEFPlant2022',
        tar_file=None,
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=True,
        index_targets=False
    ):
        """
        PC2022

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for PC2022 data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        :param index_targets: whether to index the id of each labeled image
        """

        suffix = 'train/' if train else 'val/'
        data_path = None        
        data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super(PC2022, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized PC2022')

        if index_targets:
            self.targets = []
            for sample in self.samples:
                self.targets.append(sample[1])
            self.targets = np.array(self.targets)
            self.samples = np.array(self.samples)

            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(
                    self.targets == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                logger.debug(f'num-labeled target {t} {len(indices)}')
            logger.info(f'min. labeled indices {mint}')