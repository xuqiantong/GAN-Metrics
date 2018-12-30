# GAN Metrics

This is the testing framework of sample based GAN evaluation metrics.

## Requirement
- Python  3.6.2
- torch 0.2.0_2
- torchvision 0.1.9
- pot 0.3.1
- tqdm 4.19.6
- numpy, scipy, math

## Usage
### Configuration
- Please specify all the global variables (e.g. dataset name, path, transformation parameters) in `globals.py`
- Please modify model config in `sampler/features.py` if you want to use the following to generate features:
	- ResNet with different weights
	- Models with different architectures

### Creat tasks
Use `main.py` to creat all the experiments you want to run. There are some examples in `main.py` including:
- Generate samples and features
- Compute metrics on:
	- mixtures of real/fake samples
	- samples with different size
	- sampels with transformation
- Simulate mode collapsing, mode dropping and overfitting

### Run tasks
Use `worker.py` to pick and actually excute tasks. You can sprawl as many workers as you want to run in parallel, as long as a unique id is specified for each worker: 
```
python3 worker.py --id 1
python3 worker.py --id 2
```
All the experiment results will be stored in `<g.default_repo_dir>/rlt/`.
