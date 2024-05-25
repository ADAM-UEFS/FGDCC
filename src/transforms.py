# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import PIL
from PIL import ImageFilter


import torch
import torchvision.transforms as transforms
from timm.data import create_transform


_GLOBAL_SEED = 0
logger = getLogger()


'''
    TODO: Verify this:
    MAE:
     "Our MAE works well using cropping-only augmentation, 
     either fixed-size or random-size (both having random horizontal flipping). 
     Adding color jittering degrades the results and so we do not use it in other experiments."
    
    I-JEPA (Finetuning):
     "The base learning rate is set to 10−4 and the batch size to 528. 
     We train using mixup [76] set to 0.8, cutmix [73] set to 1.0, a drop path probability of 0.25 and a weight decay set to 0.04. 
     We also use a layer decay of 0.75. Finally, we use the same rand-augment data-augmentations as MAE"

'''

def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    validation=False,
    supervised=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    logger.info('making imagenet data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort
    

    transform_list = [] 
    if validation:
        transform_list += [transforms.Resize((224,224), interpolation=PIL.Image.BICUBIC)] # to maintain same ratio w.r.t. 224 images  
        transform_list += [transforms.CenterCrop((224,224))]  
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize(normalization[0], normalization[1])]
        transform = transforms.Compose(transform_list)
        return transform
    
    if supervised:
        # -- Borrowed from MAE
        transform = create_transform(
            input_size=crop_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=normalization[0],
            std=normalization[1],
        )
        return transform

    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
