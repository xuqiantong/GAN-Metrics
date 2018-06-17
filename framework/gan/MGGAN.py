from __future__ import print_function

import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from gan.bipartite import solve
from globals import Globals, addDataInfo, getDataSet
from models import DCGAN_G
from utils import saveImage, print_prop


def MGGAN_main(opt):

    g = Globals()

    opt.workers = 2
    opt.batchSize = 64
    opt.imageSize = 64
    nc = 1 if opt.data.startswith("mnist") else 3
    opt.nz = 100
    opt.ngf = 64
    opt.ndf = 64
    opt.niter = 30
    opt.lr = 0.0002
    opt.beta1 = 0.5
    opt.cuda = True
    opt.ngpu = 1
    opt.netG = ''
    opt.netD = ''
    opt.outf = g.default_model_dir + "MGGAN/"
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

    dataset, dataloader = getDataSet(opt)

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)

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

    nloss = 200

    class _netD(nn.Module):

        def __init__(self, ngpu):
            super(_netD, self).__init__()
            self.ngpu = ngpu
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
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
            )
            self.main2 = nn.Sequential(
                # state size. (ndf*8) x 4 x 4
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, nloss, 4, 1, 0, bias=False),
                # nn.Linear(ndf*8*4*4,nloss),
                nn.Sigmoid()
            )

        def forward(self, input):
            self.feature = self.main.forward(input)
            # output=self.main2.forward(self.feature.view(input.size(0),-1))
            output = self.main2.forward(self.feature)
            return output.view(-1, 1)

    netD = _netD(ngpu)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()

    input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(opt.batchSize, nloss)
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

    real_batch = 11
    grow_speed = 5
    for epoch in range(opt.niter):
        if epoch % grow_speed == 0:
            if real_batch > 1:
                real_batch -= 1

            real_inputs = torch.FloatTensor(
                real_batch * opt.batchSize, nc, opt.imageSize, opt.imageSize)
        pointer = 0
        for i, data in enumerate(dataloader, 0):
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            if batch_size < opt.batchSize:
                continue
            pointer = pointer % real_batch + 1

            if pointer < real_batch:  # still need to fill the batch
                # copy data
                real_inputs[
                    pointer * batch_size:(pointer + 1) * batch_size].copy_(real_cpu)
                continue
            # Done collecting! Now we can collect all the feature vectors..
            input.data.resize_(real_inputs.size()).copy_(real_inputs)
            netD(input)
            true_features = netD.feature.view(
                real_inputs.size(0), -1)  # make feature a vector

            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            fake = netG(noise)
            label.data.fill_(fake_label)
            output = netD(fake.detach())
            fake_features = netD.feature.view(batch_size, -1)

            # Now we need to make a pair between pair and true.. run it as a LP
            # program...
            map = solve(fake_features.data, true_features.data)
            input.data.resize_(real_cpu.size())
            for j in range(0, batch_size):
                input.data[j].copy_(real_inputs[map[j]])

            tot_mini_batch = 10
            for mini_batch in range(0, tot_mini_batch):
                label.data.fill_(real_label)
                netD.zero_grad()
                output = netD(input)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.data.mean()

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

                print('[%d/%d][%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, opt.niter, i, len(dataloader), mini_batch, tot_mini_batch,
                         errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
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
