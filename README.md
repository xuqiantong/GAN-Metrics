# GAN Metrics

This repository provides the code for [An empirical study on evaluation metrics of generative adversarial networks](https://arxiv.org/abs/1806.07755).

## Requirement
- Python 3.6.4
- torch 0.4.0
- torchvision 0.2.1
- ot 0.4.0
- tqdm 4.19.6
- numpy, scipy, math

## Usage
We create a demo for DCGAN training as well as computing all the metrics after each epoch.
```
python3 demo_dcgan.py \
--dataset cifar10 \
--cuda \
--dataroot <data_folder> \
--outf <output_folder> \
--sampleSize 2000
```
![demo](demo.gif)
