from __future__ import print_function
import argparse
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from globals import Globals, addDataInfo, getDataSet
from utils import saveImage, print_prop
from models import WGAN_G, WGAN_D
from torch.autograd import Variable
import os


def WGAN_main(opt):
    g = Globals()

    opt.workers = 2
    opt.batchSize = 64
    opt.imageSize = 64
    opt.nz = 100
    opt.ngf = 64
    opt.ndf = 64
    opt.niter = 50
    opt.lrD = 0.00005
    opt.lrG = 0.00005
    opt.beta1 = 0.5
    opt.cuda = True
    opt.ngpu = 1
    opt.netG = ''
    opt.netD = ''
    opt.clamp_lower = -0.01
    opt.clamp_upper = 0.01
    opt.Diters = 5
    opt.n_extra_layers = 0
    opt.outf = g.default_model_dir + "WGAN/"
    opt.adam = False

    opt = addDataInfo(opt)
    opt.outf = opt.outf + opt.data + "/"
    print_prop(opt)

    if os.path.exists(opt.outf + "/mark"):
        print("Already generated before. Now exit.")
        return

    if opt.outf is None:
        opt.outf = 'samples'
    os.system('mkdir {0}'.format(opt.outf))

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    dataset, dataloader = getDataSet(opt)

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 1 if opt.data.startswith("mnist") else 3
    n_extra_layers = int(opt.n_extra_layers)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = WGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    netG.apply(weights_init)
    if opt.netG != '':  # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = WGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
    netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(
            netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(
            netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

    gen_iterations = 0
    for epoch in range(opt.niter):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = data_iter.next()
                i += 1

                # train with real
                real_cpu, _ = data
                netD.zero_grad()
                batch_size = real_cpu.size(0)

                if opt.cuda:
                    real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)

                errD_real = netD(inputv)
                errD_real.backward(one)

                # train with fake
                noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile=True)  # totally freeze netG
                fake = Variable(netG(noisev).data)
                inputv = fake
                errD_fake = netD(inputv)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            errG = netD(fake)
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                     errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            if gen_iterations % 50 == 0:
                saveImage(real_cpu, '{0}/real_samples.png'.format(opt.outf))
                fake = netG(Variable(fixed_noise, volatile=True))
                saveImage(
                    fake.data, '{0}/fake_samples_{1}.png'.format(opt.outf, gen_iterations))

        # do checkpointing
        torch.save(netG.state_dict(),
                   '{0}/netG_epoch_{1}.pth'.format(opt.outf, epoch))
        torch.save(netD.state_dict(),
                   '{0}/netD_epoch_{1}.pth'.format(opt.outf, epoch))
    with open(opt.outf + "/mark", "w") as f:
        f.write("")
