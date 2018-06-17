import shutil

import numpy as np
import torchvision.transforms as transforms

from globals import Globals
from mix import Ent
from utils import Rotate
from utils import mkdir, TMix, TGen, TExp

g = Globals()


mkdir(g.default_repo_dir)
shutil.rmtree(g.default_repo_dir + "labels", ignore_errors=True)
tasks = []


# Generate samples and features
if False:
    tasks.append(TGen('Gen', 'lsun', 'true', 9, 'resnet34'))
    tasks.append(TGen('Gen', 'lsun', 'DCGAN', 9, 'resnet34'))
    tasks.append(TGen('Gen', 'celeba', 'true', 24, 'resnet34'))
    tasks.append(TGen('Gen', 'celeba', 'DCGAN', 24, 'resnet34'))
    tasks.append(TGen('Gen', 'cifar10', 'true', 400, 'resnet34'))
    tasks.append(TGen('Gen', 'cifar10', 'WGAN', 1900, 'resnet34'))


    tasks.append(TGen('Gen', 'lsun', 'true', 9, 'vgg16'))
    tasks.append(TGen('Gen', 'lsun', 'DCGAN', 9, 'vgg16'))
    tasks.append(TGen('Gen', 'celeba', 'true', 24, 'vgg16'))
    tasks.append(TGen('Gen', 'celeba', 'DCGAN', 24, 'vgg16'))

    tasks.append(TGen('Gen', 'lsun', 'true', 9, 'inception_v3'))
    tasks.append(TGen('Gen', 'lsun', 'DCGAN', 9, 'inception_v3'))
    tasks.append(TGen('Gen', 'celeba', 'true', 24, 'inception_v3'))
    tasks.append(TGen('Gen', 'celeba', 'DCGAN', 24, 'inception_v3'))


trans = transforms.Compose([transforms.Scale(64), transforms.ToTensor(
), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans_rand = transforms.Compose([transforms.Scale(64), transforms.Pad(4), transforms.RandomCrop(
    64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans_rotate = transforms.Compose([transforms.Scale(
    64), Rotate(-1), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Metrics on different mixtures
# Only `*_smax` features are suggested to be used for `Incep`, `ModeScore` and `FID`
if False:
    for sampleSize in [2000]:
        for data in ['cifar10', 'celeba', 'lsun']:
            if data == 'celeba':
                model = 'DCGAN24'
            if data == 'lsun':
                model = 'DCGAN9'
            if data == 'cifar10':
                model = 'WGAN1900'

            for mix_ratio in np.arange(0, 1.1, 0.1):
                dataList = []
                if mix_ratio > 0:
                    dataList.append(
                        Ent(mix_ratio, data, 'true_test', trans, imageMode=0))
                if 1 - mix_ratio > 0:
                    dataList.append(
                        Ent(1 - mix_ratio, data, model, None, dup=1, imageMode=0))

                for featureType in ['resnet34_conv', 'pix']:                   
                    tasks.append(
                        TMix('Mix', sampleSize, data, featureType, dataList))

                tasks.append(
                    TMix('Incep', sampleSize, data, 'resnet34_smax', dataList))
                tasks.append(
                    TMix('ModeScore', sampleSize, data, 'resnet34_smax', dataList))
                tasks.append(
                    TMix('FID', sampleSize, data, 'resnet34_smax', dataList))


# Metrics on different number of samples
if False:
    for sampleSize in [100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for data in ['celeba', 'lsun']:
            for featureType in ['resnet34', 'vgg16', 'inception_v3']:
                for foldername in ['DCGAN24' if data == 'celeba' else 'DCGAN9', 'true_test']:
                    dataList = []
                    dataList.append(Ent(1.0, data, foldername, trans))

                    tasks.append(
                        TMix('Mix', sampleSize, data, featureType + '_conv', dataList))
                    tasks.append(
                        TMix('Incep', sampleSize, data, featureType + '_conv', dataList))
                    tasks.append(
                        TMix('ModeScore', sampleSize, data, featureType + '_conv', dataList))
                    tasks.append(
                        TMix('FID', sampleSize, data, featureType + '_conv', dataList))


# For robustness
# When using `trans_rotate` and `trans_rand`, only resnet34 is used for feature extraction.
if False:
    for sampleSize in [2000]:
        for data in ['celeba', 'lsun']:
            if data == 'celeba':
                model = 'DCGAN24'
            if data == 'lsun':
                model = 'DCGAN9'

            for mix_ratio in np.arange(0, 1.1, 0.1):
                for image_mode in [1, 2]:
                    dataList = []
                    if mix_ratio > 0:
                        dataList.append(Ent(
                            mix_ratio, data, 'true_test', trans_rotate if image_mode == 2 else trans_rand, imageMode=image_mode))
                    if 1 - mix_ratio > 0:
                        dataList.append(
                            Ent(1 - mix_ratio, data, 'true_test2', None))

                    for featureType in ['resnet34_conv', 'pix']:
                        tasks.append(
                            TMix('Mix', sampleSize, data, featureType, dataList))

                    tasks.append(
                        TMix('Incep', sampleSize, data, 'resnet34_smax', dataList))
                    tasks.append(
                        TMix('ModeScore', sampleSize, data, 'resnet34_smax', dataList))
                    tasks.append(
                        TMix('FID', sampleSize, data, 'resnet34_smax', dataList))


# Mode collapse, mode drop, overfit 
if True:
    for data in ['celeba', 'lsun']:
        for exp in ['collapse', 'drop', 'overfit']:
            tasks.append(TExp(exp, data))



print(len(tasks))
np.save(g.default_repo_dir + "tasks", tasks)
