# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)

from src.datasets.FineTuningDataset import make_GenericDataset

from src.helper import (
    configure_finetuning,
    get_classification_head,
    load_checkpoint,
    load_DC_checkpoint,
    init_model,
    init_opt,
    init_DC_opt,
    build_cache,
    VICReg
    )
from src.transforms import make_transforms
import time

# --BROUGHT fRoM MAE
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy

from src import KMeans
import faiss

# --
log_timings = True
log_freq = 50
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']

    drop_path = args['data']['drop_path']
    mixup = args['data']['mixup']
    cutmix = args['data']['cutmix']
    reprob = args['data']['reprob']
    nb_classes = args['data']['nb_classes']

    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    resume_epoch = args['data']['resume_epoch']

    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    smoothing = args['optimization']['label_smoothing']
    

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    
    load_path = None
    
    if load_model:
        load_path = '/home/rtcalumby/adam/luciano/LifeCLEFPlant2022/' +  'IN22K-vit.h.14-900e.pth.tar' #'IN1K-vit.h.14-300e.pth.tar'  #os.path.join(folder, r_file) if r_file is not None else latest_path
    
    if resume_epoch > 0:
        r_file = 'jepa-ep{}.pth.tar'.format(resume_epoch + 1)
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'Train loss'),
                           ('%.5f', 'Test loss'),
                           ('%.3f', 'Test - Acc@1'),
                           ('%.3f', 'Test - Acc@5'),
                           ('%d', 'Test time (ms)'),
                           ('%d', 'time (ms)'))
    
    stats_logger = CSVLogger(folder + '/experiment_log.csv',
                           ('%d', 'epoch'),
                           ('%.5f', 'backbone lr'),
                           ('%.5f', 'autoencoder lr'),
                           ('%.5f', 'Total Train loss'),
                           ('%.5f', 'Parent Train loss'),
                           ('%.5f', 'Parent Test loss'),
                           ('%.5f', 'Children loss'),
                           ('%.5f', 'Reconstruction loss'),
                           ('%.5f', 'K-Means loss'),
                           ('%.5f', 'Consistency loss'),
                           ('%.5f', 'VICReg loss'),
                           ('%.3f', 'Test - Acc@1'),
                           ('%.3f', 'Test - Acc@5'),
                           ('%f', 'avg_empty_clusters_per_class'),
                           ('%f', 'avg_empty_clusters_k=2'),
                           ('%f', 'avg_empty_clusters_k=3'),
                           ('%f', 'avg_empty_clusters_k=4'),
                           ('%f', 'avg_empty_clusters_k=5'),
                           ('%d', 'time (ms)'))
    
    # -- init model
    encoder, predictor, autoencoder = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)
    target_encoder = DistributedDataParallel(target_encoder, static_graph=True) # Wrap around ddp to make state dict compatible

    logger.info(autoencoder)

    training_transform = make_transforms( 
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        supervised=True,
        validation=False,
        color_jitter=color_jitter)
    
    val_transform = make_transforms( 
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        supervised=True,
        validation=True,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    _, supervised_loader_train, supervised_sampler_train = make_GenericDataset(
            transform=training_transform,
            batch_size=batch_size,
            collator=None,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=False)
    ipe = len(supervised_loader_train)
    print('Training dataset, length:', ipe*batch_size)

    # Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. 
    # This will slightly alter validation results as extra duplicate entries are added to achieve
    # equal num of samples per-process.'
    _, supervised_loader_val, supervised_sampler_val = make_GenericDataset(
            transform=val_transform,
            batch_size=batch_size,
            collator= None,
            pin_mem=pin_mem,
            training=False,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=False)
    
    ipe_val = len(supervised_loader_val)

    print('Val dataset, length:', ipe_val*batch_size)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    
    mixup_fn = None
    mixup_active = mixup > 0 or cutmix > 0.
    
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(mixup_alpha=mixup, cutmix_alpha=cutmix, label_smoothing=0.1, num_classes=nb_classes)
        print("Warning: deactivate!")
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    CEL_no_reduction = torch.nn.CrossEntropyLoss(reduction='none')

    # -- Load ImageNet weights
    if resume_epoch == 0:    
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        
    del encoder
    del predictor
    
    def save_checkpoint(epoch):
        save_dict = {
            'target_encoder': target_encoder.state_dict(),
            'classification_head': hierarchical_classifier.state_dict(),
            'opt_1': optimizer.state_dict(),
            'opt_2': AE_optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': total_loss_meter.avg,
            'parent_loss': parent_cls_loss_meter.avg,
            'subclass_loss': children_cls_loss_meter.avg,
            'reconstruction_loss': reconstruction_loss_meter.avg,
            'k_means_loss': k_means_loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    #n_blocks = 12
    #total_blocks = len(target_encoder.module.blocks)
    #for i in range((total_blocks - n_blocks), total_blocks):
    #    for p in target_encoder.module.blocks[i].parameters():
    #        p.requires_grad = True
    
    for p in target_encoder.parameters():
        p.requires_grad = True
    target_encoder = target_encoder.module 

    proj_embed_dim = 128
    VICReg_loss = VICReg(args=None, num_features=proj_embed_dim, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0) # Default coefficient values employed
    target_encoder = configure_finetuning(target_encoder, nb_classes=nb_classes, drop_path=drop_path, device=device)
    hierarchical_classifier = get_classification_head(target_encoder.pretrained_model.embed_dim, nb_classes=nb_classes, drop_path=drop_path, K_range=[2,3,4,5], proj_embed_dim=proj_embed_dim, device=device)
    
    # -- Override previously loaded optimization configs.
    # Create one optimizer that takes into account both encoder and its classifier parameters.
    optimizer, AE_optimizer, AE_scheduler, scaler, scheduler, wd_scheduler = init_DC_opt(
        encoder=target_encoder,
        classifier=hierarchical_classifier,
        autoencoder=autoencoder,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    
    autoencoder = DistributedDataParallel(autoencoder, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder, static_graph=True) # Static Graph: the set of used and unused parameters will not change during the whole training loop.    
    hierarchical_classifier = DistributedDataParallel(hierarchical_classifier, static_graph=False, find_unused_parameters=True) # Static Graph: the set of used and unused parameters will not change during the whole training loop.
    
    # TODO: ADJUST THIS later!
    if resume_epoch != 0:
        target_encoder, optimizer, scaler, start_epoch = load_DC_checkpoint(
            device=device,
            r_path=load_path,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(resume_epoch*ipe):
            scheduler.step() 
            wd_scheduler.step()

    logger.info(target_encoder)

    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = rank
    
    #resources = [faiss.StandardGpuResources() for _ in range(world_size)]
    #configs = [faiss.GpuIndexFlatConfig() for _ in range(world_size)]
    
    #configs[rank] = faiss.GpuIndexFlatConfig()
    #configs[rank].device = rank
    #configs[rank].useFloat16 = False

    K_range = [2,3,4,5]
    k_means_module = KMeans.KMeansModule(nb_classes, dimensionality=384, k_range=K_range, resources=resources, config=config)
    #k_means_module = KMeans.KMeansModule(nb_classes, dimensionality=256, k_range=K_range, resources=resources, config=configs)

    # TODO: track empty clusters per class?
    # TODO FIXME: this logging is actually implemented incorrectly. 
    # K-Means runs several iterations (until convergence) and from the way this is currently implemented
    # the code is logging empty clusters per iteration. This way the same cluster can be logged many times across iterations.
    # We will keep it as it is for now.
    empty_clusters_per_epoch = AverageMeter() # Tracks the number of empty clusters per epoch and class
    empty_clusters_per_k_per_epoch = [AverageMeter() for _ in K_range] 
    
    logger.info('Building cache...')
    cached_features_last_epoch = build_cache(data_loader=supervised_loader_train,
                                             device=device, target_encoder=target_encoder,
                                             autoencoder=autoencoder,
                                             path=root_path+'/DeepCluster/cache')
    cnt = [len(cached_features_last_epoch[key]) for key in cached_features_last_epoch.keys()]    
    assert sum(cnt) == 245897, 'Cache not compatible, corrupted or missing' # Img qtt before upsampling = 243916
    logger.info('Done.')
    logger.info('Initializing centroids...')
    k_means_module.init(resources=resources, rank=rank, cached_features=cached_features_last_epoch, config=config, device=device) # E-step
    logger.info('Done.')
    logger.info('M - Step...')
    M_losses, empty_clusters_per_epoch, empty_clusters_per_k_per_epoch = k_means_module.update(cached_features_last_epoch, device, empty_clusters_per_epoch, empty_clusters_per_k_per_epoch) # M-step
    print('Losses', M_losses)
    print('Avg no of empty clusters:', empty_clusters_per_epoch.avg)
    print('Empty clusters per K:')
    print(empty_clusters_per_k_per_epoch[0].avg,
          empty_clusters_per_k_per_epoch[1].avg,
          empty_clusters_per_k_per_epoch[2].avg,
          empty_clusters_per_k_per_epoch[3].avg)

    accum_iter = 1
    start_epoch = resume_epoch

    l2_norm = torch.nn.MSELoss()

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):

        logger.info('Epoch %d' % (epoch + 1))
        
        supervised_sampler_train.set_epoch(epoch) # Calling the set_epoch() method at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
        
        total_loss_meter = AverageMeter()
        parent_cls_loss_meter = AverageMeter()
        children_cls_loss_meter = AverageMeter()
        consistency_loss_meter = AverageMeter()
        vicreg_loss_meter = AverageMeter()
        reconstruction_loss_meter = AverageMeter()
        k_means_loss_meter = AverageMeter()

        time_meter = AverageMeter()

        target_encoder.train(True)
        hierarchical_classifier.train(True)

        cached_features = {}
        for itr, (sample, target) in enumerate(supervised_loader_train):
            
            def load_imgs():
                samples = sample.to(device, non_blocking=True)
                targets = target.to(device, non_blocking=True)
                
                # TODO: Verify how to add mixup in this hierarchical setting.     
                if mixup_fn is not None:
                    samples, targets = mixup_fn(samples, targets)
                return (samples, targets)
            
            imgs, targets = load_imgs()

            def train_step():
                _new_AE_lr = AE_scheduler.step()    
                _new_lr = scheduler.step() 
                _new_wd = wd_scheduler.step()
                
                def loss_fn(h, targets):
                    loss = criterion(h, targets)
                    loss = AllReduce.apply(loss).clone()
                    return loss 

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):

                    # Step 1. Forward into the encoder
                    h = target_encoder(imgs) 

                    h_detached = h.detach() # Detaching will prevent autograd from backpropagating the gradients from the autoencoder to the vision transformer model 
                    
                    # Step 2. Autoencoder Dimensionality Reduction  
                    reconstructed_input, bottleneck_output = autoencoder(h_detached)                     

                    reconstruction_loss = l2_norm(reconstructed_input, h_detached)

                    # Step 3. Compute K-Means assignments with disabled autocast as faiss requires float32
                    with torch.cuda.amp.autocast(enabled=False): 
                        k_means_losses, k_means_assignments = k_means_module.assign(x=bottleneck_output, y=target, resources=resources, rank=rank, device=device, cached_features=cached_features_last_epoch)  


                    # Step 4. Hierarchical Classification
                    parent_logits, child_logits, parent_proj_embeddings, child_proj_embeddings = hierarchical_classifier(h, device)

                    loss = loss_fn(parent_logits, targets)
                    parent_cls_loss_meter.update(loss)

                    ################################################## To be Replaced ###########################################################
                    #############################################################################################################################
                    classifier_selection = True
                    if classifier_selection:
                        # Model selection: Iterate through every K classifier computing the loss then select the ones with smallest values 
                        subclass_losses = []
                        for k in range(len(K_range)):
                            k_means_target = k_means_assignments[:,k,:]
                            k_means_target = k_means_target.squeeze(1)
                            subclass_loss = CEL_no_reduction(child_logits[k], k_means_target) 
                            #print('Subclass loss shape', subclass_loss.size())
                            #print('Subclass loss', subclass_loss)
                            subclass_losses.append(subclass_loss)

                        subclass_losses = torch.vstack(subclass_losses)
                        #print(subclass_losses)
                        #print(subclass_losses.size())
                        best_k_indexes = torch.argmin(subclass_losses, dim=0)

                        #print('Best K index by datapoint:', best_k_indexes)
                        #print(best_k_indexes.size())
                    #############################################################################################################################
                    #############################################################################################################################

                    # -- Setup losses
                    subclass_loss = 0
                    k_means_loss = 0
                    consistency_loss = 0

                    k_means_losses = k_means_losses.squeeze(2).transpose(0,1)

                    # -- Consistency Regularization with KL Divergence
                    size = imgs.size(0) # This allows handling cases when the number of points is different from batch size (due to drop_last=False)
                    gathered_child_embeddings = []
                    for i in range(size):
                        gathered_child_embeddings.append(child_proj_embeddings[best_k_indexes[i]][i])
                    gathered_child_embeddings = torch.stack(gathered_child_embeddings)

                    parent_probs = F.log_softmax(parent_proj_embeddings, dim=1)
                    subclass_probs = F.log_softmax(gathered_child_embeddings, dim=1)
                    consistency_loss = F.kl_div(parent_probs, subclass_probs, log_target=True,  reduction='batchmean')

                    subclass_loss = subclass_losses[best_k_indexes, torch.arange(best_k_indexes.size(0))].mean()
                    k_means_loss = k_means_losses[best_k_indexes, torch.arange(best_k_indexes.size(0))].mean()
                    
                    vicreg_loss = VICReg_loss(parent_proj_embeddings, gathered_child_embeddings)

                    #subclass_loss = AllReduce.apply(subclass_loss).clone()
                    children_cls_loss_meter.update(subclass_loss)
                    
                    consistency_loss_meter.update(consistency_loss)
                    
                    vicreg_loss_meter.update(vicreg_loss)

                    # Sum parent and subclass loss + Regularizers
                    loss += subclass_loss + consistency_loss + vicreg_loss
                    #reconstruction_loss = AllReduce.apply(reconstruction_loss).clone()
                    reconstruction_loss_meter.update(reconstruction_loss)

                    
                    # FIXME: this won't work as expected since its a constant
                    # TODO: perhaps multiplying by the gradients of the parameters from the ViT encoder.
                    # Add K-means distances term as penalty to enforce a "k-means friendly space" 
                    reconstruction_loss += 0.25 * k_means_loss 
                    '''
                        `all_reduce`: is used to perform an element-wise reduction operation (like sum, product, max, min, etc.) 
                        across all processes in a process group. 
                        The result of the reduction is stored in each tensor across all processes.
                        
                        - When you need to aggregate or synchronize values (e.g., summing gradients, averaging losses, etc.) across all processes.
                        - Typically used in model parameter synchronization during distributed training.
                    '''
                    #loss = AllReduce.apply(loss).clone()
                    
                    #k_means_loss = AllReduce.apply(k_means_loss).clone()
                    #subclass_loss = AllReduce.apply(subclass_loss).clone()
                    
                    #print('Losses')
                    #print('Parent class loss:', loss)
                    #print('Subclass loss', subclass_loss)
                    #print('K-means loss', k_means_loss)
                    #print('Autoencoder reconstruction loss', reconstruction_loss)
                # TODO: fix
                if accum_iter > 1: 
                    loss_value = loss.item()
                    reconstruction_loss_value = reconstruction_loss.item()

                    loss /= accum_iter
                    reconstruction_loss /= accum_iter
                else:
                    loss_value = loss
                    reconstruction_loss_value = reconstruction_loss 

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler(reconstruction_loss, AE_optimizer, clip_grad=1.0,
                                parameters=autoencoder.parameters(), create_graph=False, retain_graph=False,
                                update_grad=(itr + 1) % accum_iter == 0)
                    
                    scaler(loss, optimizer, clip_grad=None,
                                parameters=(list(target_encoder.parameters())+ list(hierarchical_classifier.parameters())),
                                create_graph=False, retain_graph=False,
                                update_grad=(itr + 1) % accum_iter == 0) # Scaling is only necessary when using bfloat16.   
                else:
                    reconstruction_loss.backward()
                    loss.backward()
                    optimizer.step()
                    AE_optimizer.step()

                grad_stats = grad_logger(list(target_encoder.named_parameters())+ list(hierarchical_classifier.named_parameters()))
                
                if (itr + 1) % accum_iter == 0:
                    optimizer.zero_grad()
                    AE_optimizer.zero_grad()

                return (float(loss), float(k_means_loss), _new_AE_lr, _new_lr, _new_wd, grad_stats, bottleneck_output)

            (loss, k_means_loss, ae_lr, _new_lr, _new_wd, grad_stats, bottleneck_output), etime = gpu_timer(train_step)
                       
            total_loss_meter.update(loss)
            k_means_loss_meter.update(k_means_loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d/%5d] - train_losses - Parent Class: %.4f -'
                                ' Children class: %.4f -'
                                'Autoencoder Loss (total): %.4f - Reconstruction/K-Means Loss: [%.4f / %.4f] - Consistency Loss: [%.4f]'
                                ' - VICReg Loss: [%.4f]'
                                '[wd: %.2e] [lr: %.2e] [autoencoder lr: %.2e]'
                                '[mem: %.2e] '
                                '(%.1f ms)'

                                % (epoch + 1, itr, ipe,
                                    parent_cls_loss_meter.avg,
                                    children_cls_loss_meter.avg,
                                    (reconstruction_loss_meter.avg + k_means_loss_meter.avg), reconstruction_loss_meter.avg, k_means_loss_meter.avg,
                                    consistency_loss_meter.avg,
                                    vicreg_loss_meter.avg,
                                    _new_wd,
                                    _new_lr,
                                    ae_lr,
                                    torch.cuda.max_memory_allocated() / 1024.**2,
                                    time_meter.avg))
                    
                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                        grad_stats.first_layer,
                                        grad_stats.last_layer,
                                        grad_stats.min,
                                        grad_stats.max))
            log_stats()
            bottleneck_output = bottleneck_output.to(device=torch.device('cpu'), dtype=torch.float32).detach() # Verify if apply dist.barrier
            def update_cache(cache):
                for x, y in zip(bottleneck_output, target):
                    y = y.item()
                    if not y in cache:
                        cache[y] = []                    
                    cache[y].append(x)
                return cache
            
            '''
                Warning:
                 Each device will run its own process with its own copy of the main code (including all objects that will be shared).
                 Because of that, the current epoch's cache will be updated upon different data because of DDP.
                 With this in mind we have to synchronize the update across all devices such that it is mantained consistent across all of them.
                 TODO: implement broadcasting solution.
            '''
            cached_features = update_cache(cached_features)
        # -- End of Epoch      

        if world_size > 1:
            # Convert cache to list format for gathering
            cache_list = [(key, torch.stack(value)) for key, value in cached_features.items()]

            # Gather cache lists from all processes
            all_cache_lists = [None for _ in range(world_size)]
            dist.all_gather_object(all_cache_lists, cache_list) 

            if rank == 0:
                aggregated_cache = {}
                for cache_list in all_cache_lists:
                    for key, tensor_list in cache_list:
                        if key not in aggregated_cache:
                            aggregated_cache[key] = []
                        aggregated_cache[key].extend(tensor_list)

                # Convert aggregated_cache back to the dictionary format
                aggregated_cache = {key: torch.cat(tensor_list, dim=0) for key, tensor_list in aggregated_cache.items()}
            else:
                aggregated_cache = None

            # Broadcast the aggregated cache from the root process to all other processes
            aggregated_cache = torch.distributed.broadcast_object_list(aggregated_cache, src=0)
            cached_features = {key: torch.tensor(value) for key, value in aggregated_cache}
        
        logger.info('Asserting cache length')
        # Assert everything went fine
        cnt = [len(cached_features[key]) for key in cached_features.keys()]    
        assert sum(cnt) == 245897, 'Cache not compatible, corrupted or missing'

        # TODO: same cache problem happens over here.
        # Each centroid replica is been updated according to the subset of the dataset
        # that is being handled from DDP. This means that each centroid will be updated differently if the cache 
        # is not consistent. 
        # Good news is that we only have to make the cache consistent in order to make the k-means consistent as well.
        
        # -- Perform M step on K-means module
        M_losses = k_means_module.update(cached_features, device, empty_clusters_per_epoch, empty_clusters_per_k_per_epoch)
        print('Avg no of empty clusters:', empty_clusters_per_epoch.avg)
        print('Empty clusters per K:')
        print(empty_clusters_per_k_per_epoch[0].avg,
            empty_clusters_per_k_per_epoch[1].avg,
            empty_clusters_per_k_per_epoch[2].avg,
            empty_clusters_per_k_per_epoch[3].avg)
    
        #for k in range(len(K_range)):            
        #    logger.info('Average K-Means Loss after M step: [K=',K_range[k],', value:', M_losses[k],']')
        cached_features_last_epoch = copy.deepcopy(cached_features)

        testAcc1 = AverageMeter()
        testAcc5 = AverageMeter()
        test_loss = AverageMeter()

        # Warning: Enabling distributed evaluation with an eval dataset not divisible by process number
        # will slightly alter validation results as extra duplicate entries are added to achieve equal 
        # num of samples per-process.
        @torch.no_grad()
        def evaluate():
            crossentropy = torch.nn.CrossEntropyLoss()
            target_encoder.eval()
            hierarchical_classifier.eval()              
            supervised_sampler_val.set_epoch(epoch) # -- Enable shuffling to reduce monitor bias
            
            for cnt, (samples, targets) in enumerate(supervised_loader_val):
                images = samples.to(device, non_blocking=True)
                labels = targets.to(device, non_blocking=True)
                                 
                with torch.cuda.amp.autocast():
                    output = target_encoder(images)
                    parent_logits, _, _, _ = hierarchical_classifier(output, device)                    
                    loss = crossentropy(parent_logits, labels)
                
                acc1, acc5 = accuracy(parent_logits, labels, topk=(1, 5))

                testAcc1.update(acc1)
                testAcc5.update(acc5)
                test_loss.update(loss)
        
        vtime = gpu_timer(evaluate)

        stats_logger.log(epoch + 1,
                        lr,
                        ae_lr, 
                        total_loss_meter.avg,
                        parent_cls_loss_meter.avg,
                        test_loss.avg,
                        children_cls_loss_meter.avg,
                        reconstruction_loss_meter.avg,
                        k_means_loss_meter.avg,
                        consistency_loss_meter.avg,
                        vicreg_loss_meter.avg,
                        testAcc1.avg,
                        testAcc5.avg,
                        empty_clusters_per_epoch.avg,
                        empty_clusters_per_k_per_epoch[0].avg,
                        empty_clusters_per_k_per_epoch[1].avg,
                        empty_clusters_per_k_per_epoch[2].avg,
                        empty_clusters_per_k_per_epoch[3].avg,
                        time_meter.avg)

        # -- Save Checkpoint after every epoch
        logger.info('avg. train_loss %.3f' % total_loss_meter.avg)
        logger.info('avg. test_loss %.3f avg. Accuracy@1 %.3f - avg. Accuracy@5 %.3f' % (test_loss.avg, testAcc1.avg, testAcc5.avg))
        save_checkpoint(epoch+1)
        assert not np.isnan(loss), 'loss is nan'
        logger.info('Loss %.4f' % loss)
        
        # -- Reset loggers at end of the epoch
        empty_clusters_per_epoch = AverageMeter() # Tracks the number of empty clusters per class 
        empty_clusters_per_k_per_epoch = [AverageMeter() for _ in K_range]

if __name__ == "__main__":
    main()
