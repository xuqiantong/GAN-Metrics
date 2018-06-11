import shutil

import numpy as np
import torchvision.transforms as transforms

from globals import Globals
from mix import Ent
from utils import Rotate
from utils import mkdir, TMix, TGen

g = Globals()


mkdir(g.default_repo_dir)
shutil.rmtree(g.default_repo_dir + "labels", ignore_errors=True)
tasks = []

# Generate samples and features
tasks.append(TGen('Gen', 'lsun', 'true', 9))
tasks.append(TGen('Gen', 'lsun', 'DCGAN', 9))
tasks.append(TGen('Gen', 'celeba', 'true', 24))
tasks.append(TGen('Gen', 'celeba', 'DCGAN', 24))
tasks.append(TGen('Gen', 'cifar10', 'true', 400))
tasks.append(TGen('Gen', 'cifar10', 'DCGAN', 400))
tasks.append(TGen('Gen', 'cifar10', 'WGAN', 1000))
tasks.append(TGen('Gen', 'cifar10', 'WGAN', 1900))


trans = transforms.Compose([transforms.Scale(64), transforms.ToTensor(
), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans_rand = transforms.Compose([transforms.Scale(64), transforms.Pad(4), transforms.RandomCrop(
    64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans_rotate = transforms.Compose([transforms.Scale(
    64), Rotate(-1), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Metrics on different mixtures
if True:
    for sampleSize in [2000]:
        for data in ['cifar10', 'cifar10', 'celeba', 'lsun']:
            if data == 'celeba':
                model = 'DCGAN24'
            if data == 'lsun':
                model = 'DCGAN9'
            if data == 'cifar10':
                model = 'WGAN1900'

            for featureType in ['resnet34_conv', 'resnet34_random_conv', 'pix']:
                for mix_ratio in np.arange(0, 1.1, 0.1):
                    dataList = []
                    if mix_ratio > 0:
                        dataList.append(
                            Ent(mix_ratio, data, 'true_test', trans, imageMode=0))
                    if 1 - mix_ratio > 0:
                        dataList.append(
                            Ent(1 - mix_ratio, data, model, None, dup=1, imageMode=0))

                    tasks.append(
                        TMix('Mix', sampleSize, data, featureType, dataList))
                    tasks.append(
                        TMix('Incep', sampleSize, data, 'smax', dataList))
                    tasks.append(
                        TMix('ModeScore', sampleSize, data, 'smax', dataList))
                    tasks.append(
                        TMix('FID', sampleSize, data, 'smax', dataList))


# Metrics on different number of samples
if False:
    for sampleSize in [100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for data in ['celeba', 'lsun']:
            for featureType in ['resnet34_conv']:
                for foldername in ['DCGAN24' if data == 'celeba' else 'DCGAN9', 'true_test']:
                    dataList = []
                    dataList.append(Ent(1.0, data, foldername, trans))

                    tasks.append(
                        TMix('Mix', sampleSize, data, featureType, dataList))
                    tasks.append(
                        TMix('Incep', sampleSize, data, 'smax', dataList))
                    tasks.append(
                        TMix('ModeScore', sampleSize, data, 'smax', dataList))
                    tasks.append(
                        TMix('FID', sampleSize, data, 'smax', dataList))


# For robustness
if False:
    for sampleSize in [2000]:
        for data in ['celeba', 'lsun']:
            if data == 'celeba':
                model = 'DCGAN24'
            if data == 'lsun':
                model = 'DCGAN9'

            for featureType in ['resnet34_conv', 'pix']:
                for mix_ratio in np.arange(0, 1.1, 0.1):
                    for image_mode in [1, 2]:
                        dataList = []
                        if mix_ratio > 0:
                            dataList.append(Ent(
                                mix_ratio, data, 'true_test', trans_rotate if image_mode == 2 else trans_rand, imageMode=image_mode))
                        if 1 - mix_ratio > 0:
                            dataList.append(
                                Ent(1 - mix_ratio, data, 'true_test2', None))

                        tasks.append(
                            TMix('Mix', sampleSize, data, featureType, dataList))
                        tasks.append(
                            TMix('Incep', sampleSize, data, 'smax', dataList))
                        tasks.append(
                            TMix('ModeScore', sampleSize, data, 'smax', dataList))
                        tasks.append(
                            TMix('FID', sampleSize, data, 'smax', dataList))

print(len(tasks))
np.save(g.default_repo_dir + "tasks", tasks)
