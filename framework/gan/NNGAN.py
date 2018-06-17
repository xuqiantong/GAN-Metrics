from __future__ import print_function
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import os
from models import DCGAN_G
from globals import Globals, addDataInfo, getDataSet
from metric import distance
from torch.autograd import Variable
from utils import saveImage, print_prop
import numpy as np


def NNGAN_main(opt):
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
    opt.netF = ''
    opt.netC = ''
    opt.outf = g.default_model_dir + "NNGAN/"
    opt.manualSeed = None

    opt = addDataInfo(opt)
    opt.outf = opt.outf + opt.data + "/"
    print_prop(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    if os.path.exists(opt.outf + "/mark"):
        print("Already generated before. Now exit.")
        return

    cudnn.benchmark = True

    dataset, dataloader = getDataSet(opt, needShuf=False)

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

    netG = DCGAN_G(100, nc, 64)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
        print("Load netg")
    print(netG)

    class _netFeature(nn.Module):

        def __init__(self):
            super(_netFeature, self).__init__()
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
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
            )

        def forward(self, input):
            output = self.main.forward(input).view(input.size(0), -1)
            # outputN=torch.norm(output,2,1)
            # return output/(outputN.expand_as(output))
            return output

    class _netCv(nn.Module):

        def __init__(self):
            super(_netCv, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input.view(input.size(0), 512, 4, 4)).view(-1, 1)

    netF = _netFeature()
    netF.apply(weights_init)
    print(netF)
    netC = _netCv()
    netC.apply(weights_init)
    print(netC)
    if opt.netF != '':
        netF.load_state_dict(torch.load(opt.netF))
        print("Load netf")
    if opt.netC != '':
        netC.load_state_dict(torch.load(opt.netC))
        print("Load netc")

    criterion = nn.BCELoss()

    core_batch = 64
    input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(core_batch)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        netF.cuda()
        netC.cuda()
        netG.cuda()
        criterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)

    # setup optimizer
    optimizerF = optim.Adam(netF.parameters(), lr=opt.lr,
                            betas=(opt.beta1, 0.999))
    optimizerC = optim.Adam(netC.parameters(), lr=opt.lr,
                            betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr,
                            betas=(opt.beta1, 0.999))

    core_input = Variable(torch.FloatTensor(
        core_batch, nc, opt.imageSize, opt.imageSize).cuda())

    for epoch in range(opt.niter):

        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netF.zero_grad()
            netC.zero_grad()

            noise.data.resize_(core_batch, nz, 1, 1)
            noise.data.normal_(0, 1)
            fake = netG(noise)
            label.data.resize_(core_batch).fill_(fake_label)
            fake_features = netF(fake.detach())
            output = netC(fake_features)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()

            real_cpu, _ = data
            # We only do full mini-batches, ignore the last mini-batch
            if (real_cpu.size(0) < opt.batchSize):
                print("Skip small mini batch!")
                continue
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            true_features = netF(input)
            M = distance(fake_features.data.view(fake_features.size(
                0), -1), true_features.data.view(real_cpu.size(0), -1), False)
            # get the specific neighbors of features in F_true
            _, fake_true_neighbors = torch.min(M, 1)
            unique_nn = np.unique(fake_true_neighbors.numpy()).size
            core_input.data.copy_(torch.index_select(
                real_cpu, 0, fake_true_neighbors.view(-1)))

            true_features = netF(core_input)
            output = netC(true_features)
            label.data.resize_(core_batch).fill_(real_label)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            errD = errD_real + errD_fake
            optimizerF.step()
            optimizerC.step()

            ############################
            # (2) Update G network: DCGAN
            ###########################

            netG.zero_grad()

            # fake labels are real for generator cost
            label.data.fill_(real_label)
            fake_features = netF(fake)
            output = netC(fake_features)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f D(x): %.4f D(G(z)): %.4f, %.4f unique=%d'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.data[0], D_x, D_G_z1, D_G_z2, unique_nn))

            if i % 50 == 0:
                saveImage(real_cpu[0:64], '%s/real_samples.png' % opt.outf)
                fake = netG(fixed_noise)
                saveImage(fake.data, '%s/fake_samples_epoch_%03d.png' %
                          (opt.outf, epoch))

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' %
                   (opt.outf, epoch))
        torch.save(netF.state_dict(), '%s/netF_epoch_%d.pth' %
                   (opt.outf, epoch))
        torch.save(netC.state_dict(), '%s/netC_epoch_%d.pth' %
                   (opt.outf, epoch))
    with open(opt.outf + "/mark", "w") as f:
        f.write("")
