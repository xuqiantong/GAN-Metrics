from __future__ import print_function
import argparse
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import os
from models import DCGAN_G, DCGAN_D
from globals import Globals, addDataInfo, getDataSet
from utils import saveImage, print_prop
from torch.autograd import Variable


def DCGAN_main(opt):
    g = Globals()

    opt.workers = 2
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

    dataset, dataloader = getDataSet(opt)

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

    netG = DCGAN_G(nz, nc, ngf)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = DCGAN_D(ngpu)
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
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            label.data.resize_(batch_size).fill_(real_label)

            output = netD(input)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            fake = netG(noise)
            label.data.fill_(fake_label)
            output = netD(fake.detach())
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
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                saveImage(real_cpu, '%s/real_samples.png' % opt.outf)
                fake = netG(fixed_noise)
                saveImage(fake.data, '%s/fake_samples_epoch_%03d.png' %
                          (opt.outf, epoch))

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' %
                   (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' %
                   (opt.outf, epoch))
    with open(opt.outf + "/mark", "w") as f:
        f.write("")
