from __future__ import print_function

import numpy as np
import torchvision.models as models
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dset
from globals import Globals
from utils import mkdir, lastFolder
from tqdm import tqdm
import torch.nn.functional as F
import os

def saveFeature(imgFolder, opt, model='resnet34', workers=4, batch_size=64):
    '''
        model: inception_v3, vgg13, vgg16, vgg19, resnet18, resnet34,
               resnet50, resnet101, or resnet152
    '''
    g = Globals()

    mkdir(g.default_feature_dir + opt.data)
    feature_dir = g.default_feature_dir + opt.data + "/" + lastFolder(imgFolder)
    
    conv_path = '{}_{}_conv.pth'.format(feature_dir, model)
    class_path = '{}_{}_class.pth'.format(feature_dir, model)
    smax_path = '{}_{}_smax.pth'.format(feature_dir, model)

    if (os.path.exists(conv_path) and  os.path.exists(class_path) and os.path.exists(class_path)):
        print("Feature already generated before. Now pass.")
        return

    if hasattr(opt, 'feat_model') and opt.feat_model is not None:
        model = opt.feat_model
    if model == 'vgg' or model == 'vgg16':
        vgg = models.vgg16(pretrained=True).cuda().eval()

        trans = transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        dataset = dset.ImageFolder(root=imgFolder, transform=trans)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=int(workers),
            shuffle=False)

        print('saving vgg features:')
        feature_conv, feature_smax, feature_class = [], [], []
        for img, _ in tqdm(dataloader):
            input = Variable(img.cuda(), volatile=True)
            fconv = vgg.features(input)
            fconv_out = fconv.mean(3).mean(2).squeeze()
            fconv = fconv.view(fconv.size(0), -1)
            flogit = vgg.classifier(fconv)
            fsmax = F.softmax(flogit)
            feature_conv.append(fconv_out.data.cpu())
            feature_class.append(flogit.data.cpu())
            feature_smax.append(fsmax.data.cpu())
        feature_conv = torch.cat(feature_conv, 0)
        feature_class = torch.cat(feature_class, 0)
        feature_smax = torch.cat(feature_smax, 0)

    elif model.find('resnet') >= 0:
        if model == 'resnet34_cifar':
            # Please load your own model. Example here:
            # c = torch.load(
            #     '/home/gh349/xqt/wide-resnet.pytorch/checkpoint/cifar10/gan-resnet-34.t7')
            # resnet = c['net']
            pass
            print('Using resnet34 trained on cifar10.')
            raise NotImplementedError()

        elif model == 'resnet34_random':
            # Please load your own model. Example here:
            # resnet = torch.load(
            #     '/home/gh349/xqt/wide-resnet.pytorch/checkpoint/cifar10/random_resnet34.t7')
            pass
            print('Using resnet34 with random weights.')
            raise NotImplementedError()

        else:
            resnet = getattr(models, 'resnet34')(pretrained=True)
            print('Using resnet34 with pretrained weights.')

        resnet.cuda().eval()
        resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                       resnet.maxpool, resnet.layer1,
                                       resnet.layer2, resnet.layer3, resnet.layer4)
        input = Variable(torch.FloatTensor().cuda())

        trans = transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = dset.ImageFolder(root=imgFolder, transform=trans)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=int(workers),
            shuffle=False)

        print('saving resnet features:')
        feature_conv, feature_smax, feature_class = [], [], []
        for img, _ in tqdm(dataloader):
            input = Variable(img.cuda(), volatile=True)
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

        mkdir(g.default_feature_dir)
        feature_dir = g.default_feature_dir + \
            opt.data + "/" + lastFolder(imgFolder)
        mkdir(g.default_feature_dir + opt.data)

        torch.save(feature_conv, feature_dir + '_' + model + '_conv.pth')
        torch.save(feature_class, feature_dir + '_' + model + '_class.pth')
        torch.save(feature_smax, feature_dir + '_' + model + '_smax.pth')
        return feature_conv, feature_class, feature_smax

    elif model == 'inception' or model == 'inception_v3':
        inception = models.inception_v3(
            pretrained=True, transform_input=False).cuda().eval()
        inception_feature = nn.Sequential(inception.Conv2d_1a_3x3,
                                          inception.Conv2d_2a_3x3,
                                          inception.Conv2d_2b_3x3,
                                          nn.MaxPool2d(3, 2),
                                          inception.Conv2d_3b_1x1,
                                          inception.Conv2d_4a_3x3,
                                          nn.MaxPool2d(3, 2),
                                          inception.Mixed_5b,
                                          inception.Mixed_5c,
                                          inception.Mixed_5d,
                                          inception.Mixed_6a,
                                          inception.Mixed_6b,
                                          inception.Mixed_6c,
                                          inception.Mixed_6d,
                                          inception.Mixed_7a,
                                          inception.Mixed_7b,
                                          inception.Mixed_7c,
                                          ).cuda().eval()

        trans = transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        dataset = dset.ImageFolder(root=imgFolder, transform=trans)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=int(workers),
            shuffle=False)

        print('saving resnet features:')
        feature_conv, feature_smax, feature_class = [], [], []
        for img, _ in tqdm(dataloader):
            input = Variable(img.cuda(), volatile=True)
            fconv = inception_feature(input)
            fconv = fconv.mean(3).mean(2).squeeze()
            flogit = inception.fc(fconv)
            fsmax = F.softmax(flogit)
            feature_conv.append(fconv.data.cpu())
            feature_class.append(flogit.data.cpu())
            feature_smax.append(fsmax.data.cpu())
        feature_conv = torch.cat(feature_conv, 0)
        feature_class = torch.cat(feature_class, 0)
        feature_smax = torch.cat(feature_smax, 0)

    else:
        raise NotImplementedError


    torch.save(feature_conv, '{}_{}_conv.pth'.format(feature_dir, model))
    torch.save(feature_class, '{}_{}_class.pth'.format(feature_dir, model))
    torch.save(feature_smax, '{}_{}_smax.pth'.format(feature_dir, model))
    return feature_conv, feature_class, feature_smax
