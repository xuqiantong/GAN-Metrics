# GAN Metrics

This repository provides the code for the paper [An empirical study on evaluation metrics of generative adversarial networks](https://openreview.net/pdf?id=Sy1f0e-R-).

## Usage
### Configuration
- Please specify all the global variables (e.g. dataset name, path, transformation parameters) in `globals.py`
- Please modify model config in `sampler/features.py` if you want to use:
	- ResNet with different weights
	- GANs with different architectures

### Creat tassk
Use `main.py` to creat all the experiments you want to run. There are some examples in `main.py` including:
- Generate samples and features
- Compute metrics on:
	- mixtures of real/fake samples
	- samples with different size
	- sampels with transformation

### Run tasks
Use `worker.py` to pick and actually excute tasks. You can sprawl as many workers as you want to run in parallel, as long as a unique id is specified for each worker: 
```
python3 worker.py --id 10086
```
All the experiment results will be stored in `<g.default_repo_dir>rlt/`.
