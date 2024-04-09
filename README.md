# HSViT: Horizontally Scalable Vision Transformer

[![license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2404.05196-blue)](https://arxiv.org/abs/2404.05196)

## Prerequisites

- Ubuntu 22.04
- Python 3.10 with pip

Install dependencies:

```bash
$ pip3 install -r requirements.txt
```

## Model Training

### Train HSViT

Train on CIFAR-10:

```bash
$ python3 main.py fit --config=config/hsvit_cifar10.yaml
```

Test on CIFAR-10:

```bash
$ python3 main.py test --config=config/hsvit_cifar10.yaml --ckpt_path="lightning_logs/version_0/checkpoints/epoch=0-step=38035.ckpt"
```

### Train SOTA

Train ResNet18 on CIFAR-10:

```bash
$ python3 main.py fit --config=config/sota_cifar10.yaml --model.init_args.sota_config_path="./config/huggingface/resnet-18.json"
```

## View Logs

Start a TensorBoard service based on directory of `lightning_logs`:

```bash
$ pip3 install tensorboard
$ tensorboard --logdir=lightning_logs/
```

Then go to `http://localhost:6006` on your web browser for viewing the training logs.

## Known Issues

### Bug 1

```bash
RuntimeError: received 0 items of ancdata
```
Ref: [GitHub Issue](https://github.com/fastai/fastai/issues/23#issuecomment-345091054)

Increase the `ulimit`:
```python
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
```

## Citation

If you find the code useful, please use the following BibTeX entry.

```BibTeX
@misc{xu2024hsvit,
      title={HSViT: Horizontally Scalable Vision Transformer}, 
      author={Chenhao Xu and Chang-Tsun Li and Chee Peng Lim and Douglas Creighton},
      year={2024},
      eprint={2404.05196},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```