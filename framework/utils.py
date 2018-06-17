from __future__ import print_function
import os
import torchvision.utils as vutils
import torch.utils.data as data
import torch
import PIL
import numpy as np
import torchvision.transforms as transforms


def print_prop(opt):
    print("--------------------------------------------------------------")
    print(opt.__dict__)
    print("\n--------------------------------------------------------------")


def mkdir(fname):
    try:
        os.makedirs(fname)
    except OSError:
        pass


def saveImage(img, filename, nrow=8):
    vutils.save_image(img.mul(0.5).add(0.5), filename, nrow=nrow)


class TMix(object):

    def __init__(self, mode, sampleSize, data, featureType, dataList, mixOnly=False, useModel=''):
        self.mixSize = sampleSize
        self.mode = mode
        self.data = data
        self.featureType = featureType
        self.dataList = dataList
        self.mixOnly = mixOnly  # for subclass experiment
        self.useModel = useModel


class TGen(object):

    def __init__(self, mode, data, model, epoch, feature_model='resnet34'):
        self.mode = mode
        self.data = data
        self.model = model
        self.epoch = epoch
        self.feature_model = feature_model


class TGan(object):

    def __init__(self, mode, data, model):
        self.mode = mode
        self.data = data
        self.model = model


class TExp(object):

    def __init__(self, mode, data):
        self.mode = mode
        self.data = data


def lastFolder(foldername):
    return os.path.basename(os.path.normpath(foldername))


class pickOne(object):

    def __call__(self, tensor):
        return tensor[0].view(1, 28, 28)


class Rotate(object):

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        if self.degree < 0:
            return img.rotate(np.random.randint(15) * ((-1)**(np.random.randint(2))))
        else:
            return img.rotate(self.degree)


class file_version(data.Dataset):

    def __init__(self, file, label):
        dat = torch.load(file)  # get the file

        self.len = len(dat)  # get how many data points.
        self.train = torch.FloatTensor(self.len, 1, 64, 64)
        self.label = torch.LongTensor(self.len).fill_(label)

        for i in range(0, self.len):
            img = transforms.ToPILImage()(dat[i])
            scaled = transforms.Scale(64)(img)  # scale up
            self.train[i].copy_((transforms.ToTensor()(scaled) - 0.5) * 2)

    def __getitem__(self, index):
        img, target = self.train[index], self.label[index]
        return img, target

    def __len__(self):
        return self.len


class training_version(data.Dataset):

    def __init__(self, file, labelFile):
        self.train = torch.load(file)
        self.label = torch.load(labelFile)
        self.len = len(self.train)  # get how many data points.
        for i in range(0, self.len):  # transform the imgs.
            self.train[i] = transforms.Normalize((0.1307,), (0.3081,))(
                self.train[i].view(1, -1))  # do a small transformation
        self.train = self.train.view(-1, 1, 28, 28)

    def __getitem__(self, index):
        img, target = self.train[index], self.label[index]
        return img, target

    def __len__(self):
        return self.len
