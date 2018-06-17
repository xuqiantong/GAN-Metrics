# This file randomly samples images from a folder.
# Generate three folders.

from __future__ import print_function

import argparse
import random

import torch.utils.data

import os
from globals import addDataInfo, Globals, getDataSet
from utils import mkdir, print_prop, saveImage
from .features import saveFeature
from .peek import peek

g = Globals()


def folder_sampler(opt):
    opt.workers = 2
    opt.imageSize = 64
    opt.batchSize = 600
    opt.outTrueA = 'true/'
    opt.outTrueB = 'true_test/'
    opt.outTrueC = 'true_test2/'
    opt.outf = g.default_repo_dir

    opt = addDataInfo(opt)
    assert(opt.batchSize % 3 == 0)

    print_prop(opt)
    opt.outTrueA = opt.outf + "samples/" + opt.data + "/" + opt.outTrueA
    opt.outTrueB = opt.outf + "samples/" + opt.data + "/" + opt.outTrueB
    opt.outTrueC = opt.outf + "samples/" + opt.data + "/" + opt.outTrueC
    folderList = [opt.outTrueA, opt.outTrueB, opt.outTrueC]

    if (os.path.exists(opt.outTrueC)):
        if (os.path.exists(opt.outTrueC + "/mark")):  # indeed finished
            print("Sampling already finished before. Now pass.")
            for f in folderList:
                saveFeature(f, opt, opt.feature_model)
            return
        else:
            print("Partially finished. Now rerun. ")

    mkdir(opt.outf + "samples")
    mkdir(opt.outf + "samples/" + opt.data)
    mkdir(opt.outTrueA)
    mkdir(opt.outTrueB)
    mkdir(opt.outTrueC)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset, dataloader = getDataSet(opt)

    assert(len(dataset) >= opt.sampleSize * 3)

    def giveName(iter):  # 7 digit name.
        ans = str(iter)
        return '0' * (7 - len(ans)) + ans

    iter = 0
    subfolder = -1
    splits = len(folderList)

    for i, data in enumerate(dataloader, 0):
        img, _ = data
        if i % splits == 0:
            subfolder += 1
        for j in range(0, len(img)):
            curFolder = folderList[j % splits]
            mkdir(curFolder + str(subfolder))
            if iter >= splits * opt.sampleSize:
                break
            saveImage(img[j], curFolder + str(subfolder) +
                      "/" + giveName(iter) + ".png")
            iter += 1
        if iter >= splits * opt.sampleSize:
            break

    for f in folderList:
        saveFeature(f, opt, opt.feature_model)
        peek(opt.data, os.path.relpath(f, opt.outf + "samples/" + opt.data))

    for folder in folderList:
        with open(folder + "/mark", "w") as f:
            f.write("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='celeba',
                        help='the folder to convert')
    opt = parser.parse_args()
    folder_sampler(opt)
