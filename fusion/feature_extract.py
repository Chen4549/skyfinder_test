import sys
sys.path.insert(0,'./imbalanced-regression/Skyfinder-dir')

import os
import time
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import gmean
from collections import defaultdict

import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from loss import *
from resnet import resnet50
from datasets import AgeDB

os.environ["KMP_WARNINGS"] = "FALSE"

def shot_metrics(preds, labels, train_labels, many_shot_thr=1000, low_shot_thr=100):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
    shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict

def validate(val_loader, model, train_labels=None, prefix='Val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    losses_all = []
    preds, labels = [], []

    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            outputs, _ = model(inputs)

            preds.extend(outputs.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())

            loss_mse = criterion_mse(outputs, targets)
            loss_l1 = criterion_l1(outputs, targets)
            loss_all = criterion_gmean(outputs, targets)
            losses_all.extend(loss_all.cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % 10 == 0:
                progress.display(idx)

        shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), train_labels)
        loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
        print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
        print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
              f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
        print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
              f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
        print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
              f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")

    return losses_mse.avg, losses_l1.avg, loss_gmean

def extract_visual_features_with_mapping(loader, model):
    model.eval()
    features = []
    paths = []
    
    print("===> Extracting 2048-d visual features with mapping...")
    with torch.no_grad():
        for idx, (inputs, _, _) in enumerate(tqdm(loader)):
            inputs = inputs.cuda(non_blocking=True)
            _, encoding = model(inputs) 
            
            features.append(encoding.cpu().numpy())
            # Get paths from the dataset inside the loader
            batch_indices = loader.dataset.df.index[idx * loader.batch_size : (idx + 1) * loader.batch_size]
            paths.extend(loader.dataset.df.loc[batch_indices, 'path'].values)

    return np.concatenate(features), paths

def extract(feat_dataset, reweight, lds, lds_ks, lds_sigma, fds, fds_ks, fds_sigma, checkpoint):

    checkpoint_name = checkpoint

    # Data
    print('=====> Preparing data...')
    print(f"File (.csv): {feat_dataset}")
    df = pd.read_csv(os.path.join("./imbalanced-regression/Skyfinder-dir/data", f"{feat_dataset}"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']

    df_og = pd.read_csv('./imbalanced-regression/Skyfinder-dir/data/skyfinder_30_balanced.csv')
    df_train_og = df_og[df_og['split'] == 'train']
    train_labels = df_train_og['label']

    train_dataset = AgeDB(data_dir='', df=df_train, img_size=224, split='train',
                          reweight=reweight, lds=lds, lds_kernel="gaussian", lds_ks=lds_ks, lds_sigma=lds_sigma)

    val_dataset = AgeDB(data_dir='', df=df_val, img_size=224, split='val')

    test_dataset = AgeDB(data_dir='', df=df_test, img_size=224, split='test')

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=32, pin_memory=True, drop_last=False)

    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                             num_workers=32, pin_memory=True, drop_last=False)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=32, pin_memory=True, drop_last=False)
    
    print(f"Training data size: {len(train_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    # Model
    print('\n=====> Building model...')

    model = resnet50(fds=fds, bucket_num=80, bucket_start=0,
                    start_update=0, start_smooth=1,
                    kernel="gaussian", ks=fds_ks, sigma=fds_sigma, momentum=0.9)
    model = torch.nn.DataParallel(model).cuda()

    # evaluate only
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"===> Checkpoint {checkpoint_name} loaded (epoch [{checkpoint['epoch']}]), testing...")

    print("")
    print("-"*50)
    print("===>Train set performance: ")
    validate(train_loader, model, train_labels=train_labels, prefix='Test')
    train_feat_vectors, train_paths = extract_visual_features_with_mapping(train_loader, model)

    print("")
    print("-"*50)
    print("===>Test set performance: ")
    validate(test_loader, model, train_labels=train_labels, prefix='Test')
    test_feat_vectors, test_paths = extract_visual_features_with_mapping(test_loader, model)

    return train_feat_vectors, train_paths, test_feat_vectors, test_paths

if __name__ == '__main__':
    #Example usage
    feat_dataset = "./imbalanced-regression/Skyfinder-dir/data/skyfinder_30_balanced.csv"
    reweight = "sqrt_inv"
    lds = True
    lds_ks = 5
    lds_sigma = 2
    fds = True
    fds_ks = 5
    fds_sigma = 2
    checkpoint = "./imbalanced-regression/Skyfinder-dir/checkpoint/skyfinder_30_balanced_resnet50_DIR_BASIC_lds_gau_5_2.0_fds_gau_5_2.0_0_1_0.9_adam_l1_0.001_256/ckpt.best.pth.tar"

    train_feat_vectors, train_paths, test_feat_vectors, test_paths = extract(feat_dataset, reweight, lds, lds_ks, lds_sigma, fds, fds_ks, fds_sigma, checkpoint)