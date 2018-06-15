# GAN Metrics

This repository provides the code for the paper [An empirical study on evaluation metrics of generative adversarial networks](https://openreview.net/pdf?id=Sy1f0e-R-).

## Usage
We create a demo for DCGAN training as well as computing all the metrics after each epoch.
```
python3 demo_dcgan.py \
--dataset cifar10  \
--cuda \
--dataroot <data_folder> \
--sampleSize 2000
```
 