from __future__ import print_function
import argparse
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import os
from models import DCGAN_G
from globals import Globals, addDataInfo, getDataSet
from utils import saveImage, print_prop
from torch.autograd import Variable

import numpy as np
import pdb


def DCGAN_cluster_main(opt):
    g = Globals()

    N_CLUSTER = 200
    cluster = np.load('/scratch/ys646/gan/features/celeba/clus.npy')

    opt.batchSize = 64
    opt.imageSize = 64
    opt.nz = 100
    opt.ngf = 64
    opt.ndf = 64
    opt.niter = 50
    opt.lr = 0.0002
    opt.beta1 = 0.5
    opt.cuda = True
    opt.ngpu = 1
    opt.netG = ''
    opt.netD = ''
    opt.outf = g.default_model_dir + "DCGAN/"
    opt.manualSeed = None

    opt = addDataInfo(opt)
    opt.outf = '/scratch/ys646/gan/results/'
    opt.outf = opt.outf + opt.data + "/"
    print_prop(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    if os.path.exists(opt.outf + "/mark"):
        print("Already generated before. Now exit.")
        return

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # option 1: just don't shuffle, use a counter for indices
    # this is important to keep orders fixed
    opt.workers = 1
    dataset, dataloader = getDataSet(opt, needShuf=False)
    # option 2: shuffle but use a modified dataloader that also output indices

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 1 if opt.data.startswith("mnist") else 3

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = DCGAN_G(nz + N_CLUSTER, nc, ngf)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    class _netD(nn.Module):

        def __init__(self, ngpu):
            super(_netD, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                # nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                # extra channel for input embedding
                nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
            self.emb = nn.Embedding(N_CLUSTER, opt.imageSize * opt.imageSize)

        def forward(self, input, clus_var):
            out_emb = self.emb.forward(clus_var)
            out_emb = out_emb.view(-1, 1, 64, 64)
            new_input = torch.cat([out_emb, input], 1)
            output = self.main.forward(new_input)
            return output.view(-1, 1)

    netD = _netD(ngpu)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()

    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr,
                            betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr,
                            betas=(opt.beta1, 0.999))

    for epoch in range(opt.niter):
        counter = 0
        for i, data in enumerate(dataloader, 0):
            ############################
            # (0) Get the corresponding clusters
            ###########################
            batch_size = data[1].size(0)
            clus_batch = cluster[counter:counter + batch_size]
            clus_batch = torch.from_numpy(clus_batch).long()
            counter = counter + batch_size

            clus_var = Variable(clus_batch.cuda(), requires_grad=False)
            oh = torch.FloatTensor(batch_size, N_CLUSTER)
            oh.zero_()
            oh.scatter_(1, clus_batch.view(-1, 1), 1)
            oh_var = Variable(oh.cuda(), requires_grad=False)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, _ = data
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            label.data.resize_(batch_size).fill_(real_label)

            output = netD(input, clus_var)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            # pad noise with one hot
            fake = netG(torch.cat([noise, oh_var], 1))

            label.data.fill_(fake_label)
            output = netD(fake.detach(), clus_var)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # fake labels are real for generator cost
            label.data.fill_(real_label)
            output = netD(fake, clus_var)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                saveImage(real_cpu, '%s/real_samples.png' % opt.outf)
                fake = netG(torch.cat([fixed_noise, oh_var], 1))
                saveImage(fake.data, '%s/fake_samples_epoch_%03d.png' %
                          (opt.outf, epoch))

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' %
                   (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' %
                   (opt.outf, epoch))

    with open(opt.outf + "/mark", "w") as f:
        f.write("")
