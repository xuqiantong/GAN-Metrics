from __future__ import print_function
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
from globals import Globals
from utils import mkdir, saveImage


def peek(dat, folder, force=False):
    g = Globals()
    outf = g.default_repo_dir

    mkdir(outf + "peek")
    mkdir(outf + "peek/" + dat)

    print("\nPeeking " + folder + " for " + dat)
    dir = outf + "samples/" + dat + "/" + folder
    print("dir", dir)

    if (not force) and (os.path.exists(outf + 'peek/%s/%s.png' % (dat, folder.replace("/", "_")))):
        print("Already peeked before. Now exit.")
        return

    dataset = dset.ImageFolder(root=dir,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=96,
                                             shuffle=True, num_workers=2)

    for i, data in enumerate(dataloader, 0):
        img, _ = data
        saveImage(img, outf + 'peek/%s/%s.png' %
                  (dat, folder.replace("/", "_")), nrow=12)
        break


if __name__ == '__main__':
    peek('celeba', 'true')
