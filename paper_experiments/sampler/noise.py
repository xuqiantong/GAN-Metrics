# This files reads in a generator, and sample a few images from that generator.

import argparse

import torch
from torch.autograd import Variable

import os
from globals import addDataInfo, Globals
from utils import mkdir, print_prop, saveImage
from .features import saveFeature
from .peek import peek

g = Globals()


def noise_sampler(opt):

    opt.batchSize = 64
    opt.folderSize = 600
    opt.overWrite = False
    opt.outf = g.default_repo_dir

    opt = addDataInfo(opt)

    opt.name = opt.outf + "samples/noise/true"
    print_prop(opt)

    mkdir(opt.outf + "samples")
    mkdir(opt.outf + "samples/" + opt.data)
    if (os.path.exists(opt.name)) and (not opt.overWrite):
        if (os.path.exists(opt.name + "/mark")):  # indeed finished
            print("Already generated before. Now exit.")
            return
        else:
            print("Partially finished. Now rerun. ")
    mkdir(opt.name)

    noise = Variable(torch.FloatTensor(opt.batchSize, 3, 64, 64).cuda())

    def giveName(iter):  # 7 digit name.
        ans = str(iter)
        return '0' * (7 - len(ans)) + ans

    iter = 0
    for subfolder in range(0, 1 + opt.sampleSize // opt.folderSize):
        mkdir(opt.name + "/" + str(subfolder))
        for i in range(0, 1 + opt.folderSize // opt.batchSize):
            noise.data.normal_(0, 1)
            for j in range(0, noise.data.size(0)):
                saveImage(noise.data[j], opt.name + "/" +
                          str(subfolder) + "/" + giveName(iter) + ".png")
                iter += 1
                if iter % opt.folderSize == 0:
                    break
            if iter % opt.folderSize == 0:
                break
        if iter >= opt.sampleSize:
            break
    saveFeature(opt.name, opt)
    peek(opt.data, opt.model)

    with open(opt.name + "/mark", "w") as f:
        f.write("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='noise',
                        help='the folder to convert')
    opt = parser.parse_args()
    noise_sampler(opt)
