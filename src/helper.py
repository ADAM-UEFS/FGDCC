# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch

import src.models.vision_transformer as vit
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_
import torch.nn  as nn
import torch.nn.functional as F
from torch import inf 

import src.models.autoencoder as AE

# from timm.models.layers import trunc_normal_ 
import util.lr_decay as lrd


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch

def load_FT_checkpoint(
    device,
    r_path,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return target_encoder, opt, scaler, epoch



class ParentClassifier(nn.Module):
    def __init__(self, input_dim, num_parents):
        super(ParentClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_parents)
    
    def forward(self, x):
        return self.fc(x)

class ChildClassifier(nn.Module):
    def __init__(self, input_dim, num_children):
        super(ChildClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_children)
    
    def forward(self, x):
        return self.fc(x)

class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim, num_parents, num_children_per_parent):
        super(HierarchicalClassifier, self).__init__()
        self.parent_classifier = ParentClassifier(input_dim, num_parents)
        self.child_classifiers = nn.ModuleList(
            [ChildClassifier(input_dim, num_children_per_parent) for _ in range(num_parents)]
        )
    
    def forward(self, x):
        parent_logits = self.parent_classifier(x)  # Shape (batch_size, num_parents)
        parent_probs = F.softmax(parent_logits, dim=1)  # Softmax over class dimension

        # Predict parent class
        parent_class = torch.argmax(parent_probs, dim=1)  # Argmax over class dimension

        # Use the predicted parent class to select the corresponding child classifier
        child_logits = torch.zeros(x.size(0), 5)  # Assuming each parent has exactly 5 children
        for i in range(x.size(0)):  # Iterate over each sample in the batch
            child_logits[i] = self.child_classifiers[parent_class[i]](x[i])
        
        return parent_logits, child_logits


class FinetuningModel(nn.Module):
    def __init__(self, pretrained_model, drop_path, nb_classes, K=5):
        super(FinetuningModel, self).__init__()        
        self.pretrained_model = pretrained_model
        
        self.nb_subclasses = K

        self.drop_path = drop_path
        self.nb_classes = nb_classes
        
        self.pretrained_model.drop_path = 0.2
        self.pretrained_model.drop_rate = 0.25
        
        self.head_drop = nn.Dropout(drop_path)

        self.parent_classifier = nn.Linear(self.pretrained_model.embed_dim,
                                           self.nb_classes)

        #self.intermediate_layer = nn.Linear(self.pretrained_model.embed_dim,
                                       #self.nb_subclasses)

        self.subclass_classifier = nn.Linear(self.pretrained_model.embed_dim,
                                             self.nb_classes * self.nb_subclasses)

    def forward(self, x):
        x = self.pretrained_model(x)

        x = torch.mean(x, dim=1)
        
        x = F.layer_norm(x, (x.size(-1),))  # normalize over feature-dim 

        return x

    '''
    # Another Idea:
     I guess that another possibility of doing classification hierarchically 
     is by setting up a classifier for each subclass. 
     This way, we'd have an array of linear layers with one-to-one correspondence between parent and subclasses.
     We could then make conditioned predictions by using the prediction from the parent class as index to the array of linear classifiers.
     Then we'd predict the subclass, given that we used the parent classifier prediction to select the subclass classifier.
     
     Questions:
        Which one should make more sense?
        This one seems rather insightful but it doesn't explicitly uses a product between probabilities as information as input to the criterion
        which is a bit counterintuitive. On the other hand, it seems more "semantically alligned" with, let us say, this "type of programming" 
        that we'd have one classifier for each set of classes, which it's not the further case. 

        It also seems more semantically alligned with our purpose, that we'd have separate classifiers for each subclass, such that
        it allows the model to learn explicit features for each subclass in a less fuzzier way. 

        Should we use the prediction as index, or should we use the parent (target) label directly? Does that configures cheating? Does by
        using the target directly we "disconnect" the information flow of the hierarchy? 

        TODO: Reconsider below (approach no. 3).
        Should we instead train the model to predict the cartesian product between the probabilities, instead of using the cartesian
        product as information, that is, input to a linear layer that will learns the weights for each pair/product? 
        
        I sense that this is an idea that unifies both approaches. It's scalable and reasonable.  
        Because, that is also the problem that, for our case which involves K=5, so, for problems with a small number of subclasses,
        those approaches (1 and 2), approaches that involves predicting the correct subclass are more prone to the effects randomness
        that is, for our K=5 subclasses example, the probability that the model outputs a correct class by random chance is 20%. 
        Perhaps the model wouldn't choose random strategy because it is not much successful, but to which extent this can worsen 
        the learning process by confusion? My guess is that randomness could help because it could have a strong effect in cases
        of lower values of K, and therefore, it could affect the learning process negatively. 

        Otherwise, if we try to predict a cartesian product between the two labels, in cases of a higher amount of classes, and subclasses
        the chance of randomness playing a substantial role in learning diminishes significantly.

        TODO: Rethink this approach as trying to predict cartesian products will imply the model to have to learn to predict not only the correct 
        outputs, but the incorrect ones as well? Perhaps it will turn the problem even harder. Perhaps dropping 0 values would be a viable solution. 
        Perhaps what we are calling cartesian product is not the correct name of it, perhaps the model should predict the pair of compatible labels instead.   
    '''
    def forward_classifiers(self, h): 
        
        # Output the prediction correspondent to the label provided by the dataset i.e., P(Y_p | h)
        y_parent_pred = self.parent_classifier(self.head_drop(h)) 
        
        # P(Y_s | h)
        y_subclass_pred = self.subclass_classifier(self.head_drop(h)) # Hoping that this dropout operation is performed sequentially otherwise will require a bunch of memory

        # This allows to predict the joint probability of both classes assuming that they are independent, i.e., P(Y_p, Y_s | h) = P(Y_s | h) * P(Y_p | h)
        y_subclass_pred = torch.cartesian_prod(y_subclass_pred, y_parent_pred) # But do we really want to assume independence? 
        
        return y_parent_pred, y_subclass_pred


    def forward_parent_classifier(self, x):

        x = self.pretrained_model(x)

        x = torch.mean(x, dim=1) # alternative
        
        x = F.layer_norm(x, (x.size(-1),))  # normalize over feature-dim 
        
        x = self.head_drop(x) # As in in timm.models
        
        x = self.parent_classifier(x)
        return x

    
def add_classification_head(pretrained_model, drop_path, nb_classes, device):
    model = FinetuningModel(pretrained_model, drop_path, nb_classes)

    def initialize(m):
        trunc_normal_(m.weight, std=2e-5)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

    # manually initialize fc layers
    initialize(model.parent_classifier)
    initialize(model.intermediate_layer)
    initialize(model.subclass_classifier)
    
    model.to(device)
    return model         
        

# Borrowed from MAE.
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)

    autoencoder = AE.vanilla_autoencoder()

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in autoencoder.modules():
        init_weights(m)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)


    autoencoder.to(device)
    encoder.to(device)
    predictor.to(device)
    
    #logger.info(autoencoder)
    return encoder, predictor, autoencoder

def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0 
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def init_FT_opt(
    encoder,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0 
        }
    ]

    # build optimizer with layer-wise lr decay (lrd)
    #param_groups = lrd.param_groups_lrd(encoder.pretrained_model, wd,
    #    no_weight_decay_list={'pos_embed', 'cls_token', 'dist_token'}, # decay list gathered here https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L573
    #    layer_decay=0.75
    #)

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(encoder.parameters())
    #optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = NativeScalerWithGradNormCount() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler 