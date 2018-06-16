# This files reads in a generator, and sample a few images from that generator.

import argparse

import torch
from torch.autograd import Variable

import os
from globals import addDataInfo, Globals, get_generator_loc, get_generator_model
from utils import mkdir, print_prop, saveImage
from .features import saveFeature
from .peek import peek


g = Globals()

def generator_sampler(opt):
    opt.batchSize = 64
    opt.folderSize = 600
    opt.overWrite = False
    opt.outf = g.default_repo_dir

    opt = addDataInfo(opt)
    netG = get_generator_model(opt)
    netG.load_state_dict(torch.load(get_generator_loc(opt)))
    netG.eval()

    opt.name = opt.outf + "samples/" + \
        opt.data + "/" + opt.model + str(opt.epoch)
    print_prop(opt)

    mkdir(opt.outf + "samples")
    mkdir(opt.outf + "samples/" + opt.data)
    if (os.path.exists(opt.name)) and (not opt.overWrite):
        if (os.path.exists(opt.name + "/mark")):  # indeed finished
            print("Sampling already finished before. Now pass.")
            saveFeature(opt.name, opt, opt.feature_model)
            return
        else:
            print("Partially finished. Now rerun. ")

    mkdir(opt.name)
    netG.cuda()

    noise = Variable(torch.FloatTensor(opt.batchSize, 100, 1, 1).cuda())

    def giveName(iter):  # 7 digit name.
        ans = str(iter)
        return '0' * (7 - len(ans)) + ans

    iter = 0
    for subfolder in range(0, 1 + opt.sampleSize // opt.folderSize):
        mkdir(opt.name + "/" + str(subfolder))
        for i in range(0, 1 + opt.folderSize // opt.batchSize):
            noise.data.normal_(0, 1)
            fake = netG(noise)
            for j in range(0, len(fake.data)):
                saveImage(fake.data[j], opt.name + "/" +
                          str(subfolder) + "/" + giveName(iter) + ".png")
                iter += 1
                if iter % opt.folderSize == 0 or iter >= opt.sampleSize:
                    break
            if iter % opt.folderSize == 0 or iter >= opt.sampleSize:
                break
        if iter >= opt.sampleSize:
            break

    if opt.dataset == 'mnist_s':
        print("Warning: subclass experiment.. Not saving features..")
    else:
        saveFeature(opt.name, opt, opt.feature_model)
    peek(opt.data, opt.model + str(opt.epoch))

    with open(opt.name + "/mark", "w") as f:
        f.write("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='DCGAN', help='model name')
    parser.add_argument('--epoch', type=int, default=1, help='epoch')
    parser.add_argument('--data', default='celeba',
                        help='the folder to convert')
    opt = parser.parse_args()
    generator_sampler(opt)
