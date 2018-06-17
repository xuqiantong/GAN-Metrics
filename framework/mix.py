import argparse
from globals import Globals
from utils import mkdir, saveImage
import random
import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import shutil
import torchvision.transforms as transforms
from sampler.peek import peek
import os
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

g = Globals()


class Ent():

    def __init__(self, fraction, data, folder, transform=None, dup=1, imageMode=0):
        self.fraction = fraction
        self.data = data
        self.folder = folder
        self.dup = dup
        self.imageMode = imageMode
        if transform == None:
            self.transform =\
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        else:
            self.transform = transform


def get_avg(dataloader):
    ans = torch.FloatTensor()
    tot = 0
    for i, data in enumerate(dataloader, 0):
        img, _ = data
        if i == 0:  # first read
            ans = img.sum(0)
        else:
            ans = ans + img.sum(0)
        tot += img.size(0)
    return (ans / tot)[0]


# generate samples. SingleFolder is used for subclass experiment.
def getDat(opt, dataList, outf, mixType="pix", singleFolder=False):
    size = []
    remain = opt.mixSize
    for entry in dataList:
        size.append(int(entry.fraction * opt.mixSize))  # get the correct size
        remain -= size[-1]  # update remain
    size[-1] += remain  # add the rest to the last bucket.
    assert(sum(size) == opt.mixSize)  # should add up to 1
    dat = torch.FloatTensor()
    tot = 0
    if singleFolder and os.path.exists(outf + "/mark"):
        print("Already generated before. Now exit.")
        dat = torch.load(outf + "/img.pth")
        return dat
    shutil.rmtree(outf, ignore_errors=True)
    mkdir(outf)

    if mixType == "pix":  # mix of images..
        def giveName(iter):  # 7 digit name.
            ans = str(iter)
            return '0' * (7 - len(ans)) + ans

        subfolder = -1
        subfolderSize = 600
        for entry, s in zip(dataList, size):  # should sample it one by one

            print(entry.data, entry.folder, s)
            opt.dir = g.default_repo_dir + "samples/" + entry.data + "/" + entry.folder

            opt.manualSeed = random.randint(1, 10000)  # fix seed
            random.seed(opt.manualSeed)
            torch.manual_seed(opt.manualSeed)

            # can take some transform defined by preprocess
            dataset = dset.ImageFolder(root=opt.dir, transform=entry.transform)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=96, shuffle=True, num_workers=2)

            count = 0
            # should contain more image than one need
            assert(len(dataset) >= s)
            avg_img = get_avg(dataloader)

            for i, data in enumerate(dataloader, 0):
                img, _ = data
                for candidate in img:  # add images one by one
                    # The line below is special design for dup>1 case
                    image = candidate if entry.dup == 1 else avg_img
                    for kth in range(0, entry.dup):  # duplicated images...
                        if tot == 0:
                            dat.resize_(opt.mixSize, image.size(
                                0) * image.size(1) * image.size(2))
                        if tot % subfolderSize == 0:
                            subfolder += 1
                            mkdir(outf + "/" + str(subfolder))
                        saveImage(image, outf + "/" + str(subfolder) +
                                  "/" + giveName(tot) + ".png")
                        dat[tot].fill_(0)
                        dat[tot] += image.resize_(image.nelement()) * 0.5 + 0.5
                        tot += 1
                        count += 1
                        if count == s:  # done copying
                            break
                    if count == s:  # done copying
                        break
                if count == s:  # done copying
                    break
        peek("Mix", os.path.basename(os.path.normpath(outf)), force=True)

        if singleFolder:
            torch.save(dat, outf + "/img.pth")
            torch.save([], outf + "/mark")

        return dat
    else:
        last = 0
        for entry, s in zip(dataList, size):  # should sample it one by one
            if entry.imageMode == 0:
                # no transformation, read features directly
                featureFile = g.default_feature_dir + entry.data + \
                    "/" + entry.folder + "_" + mixType + ".pth"

                featureM = torch.load(featureFile)

            else:
                # need transformation, no test
                opt.dir = g.default_repo_dir + "samples/" + entry.data + "/" + entry.folder
                dataset = dset.ImageFolder(
                    root=opt.dir, transform=entry.transform)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=96, shuffle=True, num_workers=2)

                resnet = getattr(models, 'resnet34')(pretrained=True)
                print('Using resnet34 with pretrained weights.')
                resnet.cuda().eval()
                resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                               resnet.maxpool, resnet.layer1,
                                               resnet.layer2, resnet.layer3, resnet.layer4)
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

                if mixType.find('conv') >= 0:
                    featureM = feature_conv
                elif mixType.find('smax') >= 0:
                    featureM = feature_smax
                elif mixType.find('class') >= 0:
                    featureM = feature_class
                else:
                    raise NotImplementedError

            randP = torch.randperm(len(featureM))  # random permutation
            if last == 0:
                dat.resize_(opt.mixSize, featureM.size(1))
            dat[last:last + s].copy_(featureM.index_select(0, randP[:s]))
            last += s

        torch.save(dat, outf + "/feature_" + mixType)
        return dat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixSize', type=int,
                        default=2000, help='the mix size')
    opt = parser.parse_args()
    mkdir(g.default_repo_dir + "samples/Mix")
    dataList = [
        Ent(0.8, 'mnist', 'true',
            transforms.Compose([
                transforms.Scale(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
        Ent(0.2, 'celeba', 'true', None)
    ]
    getDat(opt, dataList, g.default_repo_dir + "samples/Mix/M1")
