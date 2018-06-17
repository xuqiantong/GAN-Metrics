import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.serialization import load_lua
from torchvision.utils import make_grid, save_image
from torch import nn
from torchvision import datasets, transforms, models
from torch.multiprocessing import Pool
import torchvision.datasets as dset

import matplotlib.pyplot as plt
from collections import Counter
import cv2
import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
from scipy import linalg
from sklearn.cluster import KMeans
import os

import globals
from metric import distance, wasserstein, knn, mmd, inception_score, mode_score, fid

g = globals.Globals()


## ResNet feature generator
device_id = 0

def get_features(imgs, batch_size=100):
    resnet = models.resnet34(pretrained=True).cuda(device_id).eval()
    resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                   resnet.maxpool, resnet.layer1,
                                   resnet.layer2, resnet.layer3, resnet.layer4).cuda(device_id).eval()
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(64),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    feature_conv, feature_smax, feature_class = [], [], []
    for batch in tqdm(imgs.split(batch_size)):
        batch = torch.stack(list(map(trans, batch)), 0)
        input = Variable(batch.cuda(device_id))
        fconv = resnet_feature(input)
        fconv = fconv.mean(3).mean(2).squeeze()
        flogit = resnet.fc(fconv)
        fsmax = F.softmax(flogit)
        feature_conv.append(fconv.data.cpu())
        feature_class.append(flogit.data.cpu())
        feature_smax.append(fsmax.data.cpu())
    feature_conv = torch.cat(feature_conv, 0)
    feature_class = torch.cat(feature_class, 0)
    feature_smax = torch.cat(feature_smax, 0)
    return feature_conv, feature_class, feature_smax


# Mode collapse experiments
def collapse_exp_1(r_feat_val, r_feat, c_feat, pred):
    # emd, mmd, acc_t, acc_f
    n_mode = c_feat.size(0)
    c_feat_repeat = c_feat[pred]
    scores = np.zeros((n_mode, 4))
    t_feat = r_feat.clone()
    index = torch.arange(0, 2000).long()
    collapsed_order = torch.randperm(n_mode).long()
    Mxx = distance(r_feat_val, r_feat_val, sqrt=False)
    
    for i in range(n_mode):
        # Compute Score
        Mxy = distance(r_feat_val, t_feat, sqrt=False)
        Myy = distance(t_feat, t_feat, sqrt=False)
        scores[i, 0] = wasserstein(Mxy, True)
        scores[i, 1] = mmd(Mxx, Mxy, Myy, 1)
        s = knn(Mxx, Mxy, Myy, 1, True)
        scores[i, 2], scores[i, 3] = s.acc_t, s.acc_f
        
        # Do collapse 
        c = collapsed_order[i]
        cidx = index[pred.eq(c)]
        t_feat[cidx] = c_feat_repeat[cidx]
        
    return scores

def collapse_exp_2(r_feat_val, r_feat, c_feat, pred):
    # incep_score, mode_score, fid
    n_mode = c_feat.size(0)
    c_feat_repeat = c_feat[pred]
    scores = np.zeros((n_mode, 3))
    t_feat = r_feat.clone()
    index = torch.arange(0, 2000).long()
    collapsed_order = torch.randperm(n_mode).long()
    Mxx = distance(r_feat_val, r_feat_val, sqrt=False)
    
    for i in range(n_mode):
        # Compute Score
        Mxy = distance(r_feat_val, t_feat, sqrt=False)
        Myy = distance(t_feat, t_feat, sqrt=False)
        scores[i, 0] = inception_score(t_feat)
        scores[i, 1] = mode_score(t_feat, r_feat_val)
        scores[i, 2] = fid(t_feat, r_feat_val)
        
        # Do collapse 
        c = collapsed_order[i]
        cidx = index[pred.eq(c)]
        t_feat[cidx] = c_feat_repeat[cidx]
        
    return scores


# Mode drop experiments
def drop_exp_1(r_feat_val, r_feat_train, pred):
    # emd, mmd, acc_t, acc_f
    n_mode = len(Counter(pred))
    scores = np.zeros((n_mode, 4))
    t_feat = r_feat_train.clone()
    collapsed_order = torch.randperm(n_mode).long()
    index = torch.arange(0, r_feat_train.size(0)).long()
    collapsed = torch.zeros(r_feat_train.size(0)).byte()
    Mxx = distance(r_feat_val, r_feat_val, sqrt=True)
    
    for i in range(n_mode):
        # Compute Score
        Mxy = distance(r_feat_val, t_feat, sqrt=True)
        Myy = distance(t_feat, t_feat, sqrt=True)
        scores[i, 0] = wasserstein(Mxy, False)
        scores[i, 1] = mmd(Mxx, Mxy, Myy, 1)
        s = knn(Mxx, Mxy, Myy, 1, True)
        scores[i, 2], scores[i, 3] = s.acc_t, s.acc_f
        
        # Do drop -- fill dropped slots with remaining samples
        c = collapsed_order[i]
        collapsed[pred.eq(c)] = 1
        cidx = index[collapsed.eq(1)]
        ncidx = index[collapsed.ne(1)]
        if ncidx.dim() == 0 or cidx.dim() == 0 or ncidx.size(0) == 0:
            continue
        for j in cidx:
            copy_idx = np.random.randint(0, ncidx.size(0))
            t_feat[j] = t_feat[ncidx[copy_idx]]
            
    return scores

def drop_exp_2(r_feat_val, r_feat_train, pred):
    # incep_score, mode_score, fid
    n_mode = len(Counter(pred))
    scores = np.zeros((n_mode, 3))
    t_feat = r_feat_train.clone()
    collapsed_order = torch.randperm(n_mode).long()
    index = torch.arange(0, r_feat_train.size(0)).long()
    collapsed = torch.zeros(r_feat_train.size(0)).byte()
    Mxx = distance(r_feat_val, r_feat_val, sqrt=True)
    
    for i in range(n_mode):
        # Compute Score
        Mxy = distance(r_feat_val, t_feat, sqrt=True)
        Myy = distance(t_feat, t_feat, sqrt=True)
        scores[i, 0] = inception_score(t_feat)
        scores[i, 1] = mode_score(t_feat, r_feat_val)
        scores[i, 2] = fid(t_feat, r_feat_val)
        
        # Do drop -- fill dropped slots with remaining samples
        c = collapsed_order[i]
        collapsed[pred.eq(c)] = 1
        cidx = index[collapsed.eq(1)]
        ncidx = index[collapsed.ne(1)]
        if ncidx.dim() == 0 or cidx.dim() == 0 or ncidx.size(0) == 0:
            continue
        for j in cidx:
            copy_idx = np.random.randint(0, ncidx.size(0))
            t_feat[j] = t_feat[ncidx[copy_idx]]
            
    return scores


# Overfitting experiments
def overfit_exp_1(r_feat_val, r_feat_train, step=200):
    # incep_score, mode_score, fid
    n_mode = r_feat_train.size(0) // step
    scores = np.zeros((n_mode+1, 4))
    t_feat = r_feat_train.clone()
    collapsed_order = torch.randperm(n_mode).long()
    index = torch.arange(0, r_feat_train.size(0)).long()
    collapsed = torch.zeros(r_feat_train.size(0)).byte()
    Mxx = distance(r_feat_val, r_feat_val, sqrt=True)
    
    for i in range(n_mode+1):
        # Compute Score
        Mxy = distance(r_feat_val, t_feat, sqrt=True)
        Myy = distance(t_feat, t_feat, sqrt=True)
        scores[i, 0] = wasserstein(Mxy, False)
        scores[i, 1] = mmd(Mxx, Mxy, Myy, 1)
        s = knn(Mxx, Mxy, Myy, 1, True)
        scores[i, 2], scores[i, 3] = s.acc_t, s.acc_f
        
        # Copy samples so as to overfit
        if i == n_mode: break
        t_feat[i*step:(i+1)*step] = r_feat_val[i*step:(i+1)*step]
    
    return scores

def overfit_exp_2(r_feat_val, r_feat_train, step=200):
    # incep_score, mode_score, fid
    n_mode = r_feat_train.size(0) // step
    scores = np.zeros((n_mode+1, 3))
    t_feat = r_feat_train.clone()
    collapsed_order = torch.randperm(n_mode).long()
    index = torch.arange(0, r_feat_train.size(0)).long()
    collapsed = torch.zeros(r_feat_train.size(0)).byte()
    Mxx = distance(r_feat_val, r_feat_val, sqrt=True)
    
    for i in range(n_mode+1):
        # Compute Score
        Mxy = distance(r_feat_val, t_feat, sqrt=True)
        Myy = distance(t_feat, t_feat, sqrt=True)
        scores[i, 0] = inception_score(t_feat)
        scores[i, 1] = mode_score(t_feat, r_feat_val)
        scores[i, 2] = fid(t_feat, r_feat_val)
        
        # Copy samples so as to overfit
        if i == n_mode: break
        t_feat[i*step:(i+1)*step] = r_feat_val[i*step:(i+1)*step]
    
    return scores


# Datasets
def get_dataset(dataset_name):
    if dataset_name == 'celeba':
        celeba = dset.ImageFolder(root=g.default_data_dir,
                                   transform=transforms.Compose([
                                       transforms.CenterCrop(138),
                                       transforms.Scale(64),
                                       transforms.ToTensor(),
                                   ]))
        return celeba

    if dataset_name == 'lsun':
        lsun = dset.LSUN(db_path=g.default_data_dir+'lsun/', classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Scale(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        return lsun


# Let's run
def prepare(dataset):
    real_idx = torch.randperm(len(dataset)).long()
    r_imgs = torch.stack([dataset[i][0] for i in tqdm(real_idx[:2000])], 0)
    r2_imgs = torch.stack([dataset[i][0] for i in tqdm(real_idx[2000:4000])], 0)
    kmeans = KMeans(n_clusters=50, n_jobs=12)
    X = r_imgs.view(2000, -1).numpy()
    kmeans.fit(X)
    centers = torch.from_numpy(kmeans.cluster_centers_).view(-1, 3, 64, 64).float()
    r_feat = get_features(r_imgs)
    r2_feat = get_features(r2_imgs)
    c_feat = get_features(centers)
    pred = distance(r_imgs, centers, False).min(1)[1].squeeze_()

    return r_imgs, r2_imgs, centers, r_feat, r2_feat, c_feat, pred

def run_overfit_exp(dataset_name):
    dataset = get_dataset(dataset_name)
    r_imgs, r2_imgs, centers, r_feat, r2_feat, c_feat, pred = prepare(dataset)

    for i in range(5):
        path = '{}/rlt/overfit/{}_scores_{}.pth'.format(g.default_repo_dir, dataset_name, i)
        if not os.path.exists(path):
            pix_scores = overfit_exp_1(r2_imgs, r_imgs)
            conv_scores = overfit_exp_1(r2_feat[0], r_feat[0])
            smax_scores = overfit_exp_2(r2_feat[2], r_feat[2])
            torch.save([pix_scores, conv_scores, smax_scores], path)

def run_collapse_exp(dataset_name):
    dataset = get_dataset(dataset_name)
    r_imgs, r2_imgs, centers, r_feat, r2_feat, c_feat, pred = prepare(dataset)

    for i in range(5):
        path = '{}/rlt/collapse/{}_scores_{}.pth'.format(g.default_repo_dir, dataset_name, i)
        if not os.path.exists(path):
            pix_scores = collapse_exp_1(r2_imgs, r_imgs, centers, pred)
            conv_scores = collapse_exp_1(r2_feat[0], r_feat[0], c_feat[0], pred)
            smax_scores = collapse_exp_2(r2_feat[2], r_feat[2], c_feat[2], pred)
            torch.save([pix_scores, conv_scores, smax_scores], path)

def run_drop_exp(dataset_name):
    dataset = get_dataset(dataset_name)
    r_imgs, r2_imgs, centers, r_feat, r2_feat, c_feat, pred = prepare(dataset)

    for i in range(5):
        path = '{}/rlt/drop/{}_scores_{}.pth'.format(g.default_repo_dir, dataset_name, i)
        if not os.path.exists(path):
            pix_scores = drop_exp_1(r2_imgs, r_imgs, pred)
            conv_scores = drop_exp_1(r2_feat[0], r_feat[0], pred)
            smax_scores = drop_exp_2(r2_feat[2], r_feat[2], pred)
            torch.save([pix_scores, conv_scores, smax_scores], path)
