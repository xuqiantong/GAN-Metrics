# this is to sample some subclasses, i.e., in MNIST1, MNIST2 or so.
# Data name is different from its mother. Say MNIST

from __future__ import print_function

import argparse

import os
from globals import addDataInfo, Globals, getDataSet
from utils import mkdir, print_prop, saveImage
from .features import saveFeature
from .peek import peek
import torch

g = Globals()


def subclass_sampler(opt):
    assert(opt.data == 'mnist')
    opt.workers = 2
    opt.imageSize = 64
    opt.batchSize = 600
    opt.outf = g.default_repo_dir

    opt = addDataInfo(opt)
    print_prop(opt)

    saved = []
    for i in range(0, 10):
        saved.append([])
    opt.outTrue9 = opt.outf + "samples/" + opt.data + "9/true"
    if (os.path.exists(opt.outTrue9)):
        if (os.path.exists(opt.outTrue9 + "/mark")):  # indeed finished
            print("Already generated before. Now exit.")
            return
        else:
            print("Partially finished. Now rerun. ")

    dataset, dataloader = getDataSet(opt)

    for batch_idx, (data, target) in enumerate(dataloader):
        for d, t in zip(data, target):
            saved[t].append(d * 0.3081 + 0.1307)

    opt.data_pre = opt.data
    for i in range(0, 10):

        mkdir(opt.outf + "samples")
        mkdir(opt.outf + "samples/" + opt.data_pre + str(i))
        curFolder = opt.outf + "samples/" + opt.data_pre + str(i) + "/true/"
        mkdir(curFolder)

        def giveName(iter):  # 7 digit name.
            ans = str(iter)
            return '0' * (7 - len(ans)) + ans

        subfolder = -1
        for s in range(0, len(saved[i])):
            if s % 600 == 0:
                subfolder += 1
                mkdir(curFolder + str(subfolder))
            saveImage(saved[i][s] * 2 - 1, curFolder +
                      str(subfolder) + "/" + giveName(s) + ".png")

        peek(opt.data, 'true', True)
        torch.save(saved[i], curFolder + "dat.pth")

        with open(curFolder + "/mark", "w") as f:
            f.write("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='mnist',
                        help='the folder to convert')
    opt = parser.parse_args()
    subclass_sampler(opt)
