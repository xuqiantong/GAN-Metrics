import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from models import MDGAN_G, DCGAN_G, WGAN_G, MDGAN_D, DCGAN_D, WGAN_D
from utils import file_version


def get_hostname():
    with open("/etc/hostname") as f:
        hostname = f.read()
    hostname = hostname.split('\n')[0]
    return hostname


class Globals():

    def __init__(self):
        self.default_data = 'lsun'
        self.default_root_dir = '/home/gh349/xqt/new_gan_exps/'
        self.default_feature_dir = self.default_root_dir + 'features/'
        self.default_repo_dir = self.default_root_dir + 'repo/'
        self.default_data_dir = '/scratch/gh349/'
        self.default_model_dir = '/home/gh349/gan/data/models/'
        self.default_ground_truth_folder = 'true'

g = Globals()


def addDataInfo(opt):
    if opt.data == 'mnist':
        opt.sampleSize = 20000
        opt.dataset = 'mnist'
        opt.dataroot = g.default_data_dir + 'MNIST'

    elif opt.data == 'celeba':
        opt.sampleSize = 20000
        opt.dataset = 'celeba'
        opt.dataroot = g.default_data_dir + 'celeba'

    elif opt.data == 'cifar10':
        opt.sampleSize = 16200
        opt.dataset = 'cifar10'
        opt.dataroot = g.default_data_dir

    elif opt.data == 'lsun':
        opt.sampleSize = 20000
        opt.dataset = 'lsun'
        opt.dataroot = g.default_data_dir + 'lsun'

    elif opt.data == 'noise':
        # shouldn't consider sample size, should pick all subclasses.
        opt.sampleSize = 20000

    elif opt.data.startswith("mnist"):  # pick subclasses
        opt.dataset = 'mnist_s'
        opt.dataroot = g.default_repo_dir + "samples/" + opt.data + "/true"
        ds = dset.ImageFolder(root=opt.dataroot)
        # shouldn't consider sample size, should pick all subclasses.
        opt.sampleSize = len(ds)

    return opt


def get_generator_loc(opt):
    if opt.model == "DCGAN":
        return g.default_model_dir + "DCGAN/" + opt.data + "/netG_epoch_" + str(opt.epoch) + ".pth"
    elif opt.model == "WGAN":
        return g.default_model_dir + "WGAN/" + opt.data + "/netG_epoch_" + str(opt.epoch) + ".pth"
    elif opt.model == "MDGAN":
        return g.default_model_dir + "MDGAN/" + opt.data + "/netG_epoch_" + str(opt.epoch) + ".pth"
    elif opt.model == "NNGAN":
        return g.default_model_dir + "NNGAN/" + opt.data + "/netG_epoch_" + str(opt.epoch) + ".pth"
    elif opt.model == "MGGAN":
        return g.default_model_dir + "MGGAN/" + opt.data + "/netG_epoch_" + str(opt.epoch) + ".pth"
    assert(False)


def get_discrim_loc(opt):
    if opt.model == "DCGAN":
        return g.default_model_dir + "DCGAN/" + opt.data + "/netD_epoch_" + str(opt.epoch) + ".pth"
    elif opt.model == "WGAN":
        return g.default_model_dir + "WGAN/" + opt.data + "/netD_epoch_" + str(opt.epoch) + ".pth"
    elif opt.model == "MDGAN":
        return g.default_model_dir + "MDGAN/" + opt.data + "/netD_epoch_" + str(opt.epoch) + ".pth"
    elif opt.model == "NNGAN":
        return g.default_model_dir + "NNGAN/" + opt.data + "/netD_epoch_" + str(opt.epoch) + ".pth"
    elif opt.model == "MGGAN":
        return g.default_model_dir + "MGGAN/" + opt.data + "/netD_epoch_" + str(opt.epoch) + ".pth"
    assert(False)


def get_generator_model(opt):
    nc = 1 if opt.data.startswith("mnist") else 3
    if opt.model == "DCGAN":
        return DCGAN_G(100, nc, 64)
    elif opt.model == "WGAN":
        return WGAN_G(64, 100, nc, 64, 1)
    elif opt.model == "MDGAN":
        return MDGAN_G(nc)
    elif opt.model == "NNGAN":
        return DCGAN_G(100, nc, 64)
    elif opt.model == "MGGAN":
        return DCGAN_G(100, nc, 64)
    assert(False)


def get_discrim_model(opt):
    nc = 1 if opt.data.startswith("mnist") else 3
    if opt.model == "DCGAN":
        return DCGAN_D()
    elif opt.model == "WGAN":
        return WGAN_D(64, 100, nc, 64, 1)
    elif opt.model == "MDGAN":
        return MDGAN_D()
    assert(False)


def getDataSet(opt, needShuf=True):
    if opt.dataset in ['imagenet', 'folder', 'lfw', 'celeba']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.CenterCrop(138),
                                       transforms.Scale(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    elif opt.dataset == 'mnist_s':
        dataset = file_version(opt.dataroot + '/dat.pth', int(opt.data[-1]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=needShuf, num_workers=int(opt.workers))
    return dataset, dataloader
