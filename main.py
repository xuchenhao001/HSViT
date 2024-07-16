try:
    import resource
except ModuleNotFoundError:
    # Error handling
    pass
import sys

import lightning as L
from lightning.pytorch.cli import LightningCLI
import torch

from trainers import *  # noqa: import lightning modules
import datasets.data_modules  # noqa: import lightning data modules


def cli_main():
    # https://github.com/Lightning-AI/pytorch-lightning/issues/12997
    torch.set_float32_matmul_precision("high")  # for tensor core acceleration
    if "resource" in sys.modules:
        # https://github.com/fastai/fastai/issues/23#issuecomment-345091054
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))  # increase `ulimit` to 4096

    LightningCLI(L.LightningModule, L.LightningDataModule, subclass_mode_model=True, subclass_mode_data=True)


if __name__ == "__main__":
    cli_main()
