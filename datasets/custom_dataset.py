import os
from typing import Tuple, List, Optional, Callable

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class CustomImageDataset(VisionDataset):
    def __init__(self, root: str = None,  transform: Optional[Callable] = None):
        super().__init__(root=root, transform=transform)
        self.samples = self.make_dataset(self.root)

    def is_image_file(self, filename: str, extensions: Tuple = IMG_EXTENSIONS) -> bool:
        return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

    def make_dataset(self, directory: str) -> List[str]:
        directory = os.path.expanduser(directory)

        instances = []
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if self.is_image_file(path):
                    instances.append(path)
        return instances

    def __getitem__(self, index):
        path = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)
