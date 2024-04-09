from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, Food101
from torchvision.transforms import v2

from datasets.custom_dataset import CustomImageDataset
from datasets.tiny_imagenet.tiny_imagenet import TinyImageNet


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int, num_workers: int, image_size: int):
        super().__init__()
        root_dir = Path(root_dir)
        assert root_dir.exists(), f"provided path {root_dir} does not exist"

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.dataset_train, self.dataset_test = None, None

    def setup(self, stage: str):
        transform_train = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.RandAugment(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

        transform_test = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

        if stage == "fit":
            self.dataset_train = CIFAR10(root=str(self.root_dir), train=True, transform=transform_train)
        self.dataset_test = CIFAR10(root=str(self.root_dir), train=False, transform=transform_test)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def collate_fn(self, batch):
        cutmix = v2.CutMix(num_classes=10)
        mixup = v2.MixUp(num_classes=10)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        return cutmix_or_mixup(*default_collate(batch))


class CIFAR100DataModule(L.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int, num_workers: int, image_size: int):
        super().__init__()
        root_dir = Path(root_dir)
        assert root_dir.exists(), f"provided path {root_dir} does not exist"

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.dataset_train, self.dataset_test = None, None

    def setup(self, stage: str):
        transform_train = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.RandAugment(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])

        transform_test = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])

        if stage == "fit":
            self.dataset_train = CIFAR100(root=str(self.root_dir), train=True, transform=transform_train)
        self.dataset_test = CIFAR100(root=str(self.root_dir), train=False, transform=transform_test)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def collate_fn(self, batch):
        cutmix = v2.CutMix(num_classes=100)
        mixup = v2.MixUp(num_classes=100)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        return cutmix_or_mixup(*default_collate(batch))


class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int, num_workers: int, image_size: int):
        super().__init__()
        root_dir = Path(root_dir)
        assert root_dir.exists(), f"provided path {root_dir} does not exist"

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.dataset_train, self.dataset_test = None, None

    def setup(self, stage: str):
        transform_train = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.RandAugment(),
            v2.ToDtype(torch.float, scale=True),
        ])

        transform_test = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.ToDtype(torch.float, scale=True),
        ])

        if stage == "fit":
            self.dataset_train = FashionMNIST(root=str(self.root_dir), train=True, transform=transform_train)
        self.dataset_test = FashionMNIST(root=str(self.root_dir), train=False, transform=transform_test)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def collate_fn(self, batch):
        cutmix = v2.CutMix(num_classes=10)
        mixup = v2.MixUp(num_classes=10)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        return cutmix_or_mixup(*default_collate(batch))


class Food101DataModule(L.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int, num_workers: int, image_size: int):
        super().__init__()
        root_dir = Path(root_dir)
        assert root_dir.exists(), f"provided path {root_dir} does not exist"

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.dataset_train, self.dataset_test = None, None

    def setup(self, stage: str):
        transform_train = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.RandAugment(),
            v2.ToDtype(torch.float, scale=True),
        ])

        transform_test = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.ToDtype(torch.float, scale=True),
        ])

        if stage == "fit":
            self.dataset_train = Food101(root=str(self.root_dir), split="train", transform=transform_train)
        self.dataset_test = Food101(root=str(self.root_dir), split="test", transform=transform_test)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def collate_fn(self, batch):
        cutmix = v2.CutMix(num_classes=101)
        mixup = v2.MixUp(num_classes=101)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        return cutmix_or_mixup(*default_collate(batch))


class TinyImageNetDataModule(L.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int, num_workers: int, image_size: int):
        super().__init__()
        root_dir = Path(root_dir)
        assert root_dir.exists(), f"provided path {root_dir} does not exist"

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.dataset_train, self.dataset_test = None, None

    def setup(self, stage: str):
        transform_train = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.RandAugment(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if stage == "fit":
            self.dataset_train = TinyImageNet(root=str(self.root_dir), train=True, transform=transform_train)
        self.dataset_test = TinyImageNet(root=str(self.root_dir), train=False, transform=transform_test)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def collate_fn(self, batch):
        cutmix = v2.CutMix(num_classes=200)
        mixup = v2.MixUp(num_classes=200)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        return cutmix_or_mixup(*default_collate(batch))


class CustomDataModule(L.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int, num_workers: int, image_size: int):
        super().__init__()
        root_dir = Path(root_dir)
        assert root_dir.exists(), f"provided path {root_dir} does not exist"

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.dataset_test = None

    def setup(self, stage: str):
        transform_test = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            # v2.RandAugment(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.dataset_test = CustomImageDataset(root=str(self.root_dir), transform=transform_test)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
