# ref https://github.com/towzeur/vision/commit/a67feb569361f440fd48ed492183de8bd8f6b585
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from tqdm import tqdm

import numpy as np
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, extract_archive
from torchvision.datasets.vision import VisionDataset


class TinyImageNet(VisionDataset):
    """`TinyImageNet <https://www.kaggle.com/c/tiny-imagenet>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``tiny-imagenet-200`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from val set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "tiny-imagenet-200"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    zip_md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    train_list = [
        ["train_data", "e87bb8c13ebfecd6e9bd44b03c33108b"],
        ["train_targets", "8559fd56eaf2c3d0425014cab5574fe0"],
        ["train_bboxes", "613dd1e1ad67c6d24a11935472c0ba63"]
    ]

    test_list = [
        ["test_data", "feb4b1c955cfde3a51181fca197144cd"],
        ["test_targets", "3698446f5b44a083f3f1e18cb8382da1"],
        ["test_bboxes", "1de94c9b303983505c44e76803b396e8"]
    ]

    NUM_CLASSES = 200
    INPUT_SIZE = 64
    TRAIN_SIZE = 100000
    TEST_SIZE = 10000

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.split = 'train' if train else 'test'

        self.base_dir = Path(root) / self.base_folder
        self.zip_file = Path(root) / self.filename
        self.split_dir = self.base_dir / ('train' if train else 'val')
        self.npy_dir = self.base_dir / 'npy'

        # download zip file
        if download:
            self.download()
        if not self._check_zip_integrity():
            raise RuntimeError(f"`{self.filename}` not found or corrupted. You can use download=True to download it")
        if not self.base_dir.exists():
            print("Archive not extracted. Extracting...")
            extract_archive(self.zip_file, self.base_dir)

        self._load_meta()
        self._load_or_construct_files()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _load_or_construct_files(self) -> None:
        '''
        load if files exist, otherwise construct them
        numpy files for quick load and access
        '''
        self.data_file = self.npy_dir / f'{self.split}_data.npy'
        self.targets_file = self.npy_dir / f'{self.split}_targets.npy'
        self.bboxes_file = self.npy_dir / f'{self.split}_bboxes.npy'

        if self._check_integrity():
            print("Numpy files already constructed and verified")
            # load numpy files
            self.data = np.load(self.data_file)
            self.targets = np.load(self.targets_file)
            self.bboxes = np.load(self.bboxes_file, allow_pickle=True)
        else:
            print("Numpy files not found or corrupted. Constructing...")
            self._parse_dataset()

            # save for quick access:
            self.npy_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.data_file, self.data)
            np.save(self.targets_file, self.targets)
            np.save(self.bboxes_file, self.bboxes)

    def _load_meta(self) -> None:
        # _classes = [n02124075,...,n02504458]
        with (self.base_dir / 'wnids.txt').open() as file:
            self._classes = [x.strip() for x in file.readlines()]

        self.class_to_idx = {name: i for i, name in enumerate(self._classes)}
        self.idx_to_class = {i: name for i, name in enumerate(self._classes)}

        # classes = ['Egyptian cat',...,'African elephant, Loxodonta africana']
        self.classes = [None] * len(self._classes)
        with (self.base_dir / 'words.txt').open() as file:
            for line in file:
                name, readable_name = line.rstrip().split('\t')
                if name in self.class_to_idx:
                    self.classes[self.class_to_idx[name]] = readable_name

    def _check_integrity(self) -> bool:
        split_list = self.train_list if self.train else self.test_list
        for filename, md5 in split_list:
            fpath = self.npy_dir / (filename + '.npy')
            if not check_integrity(fpath, md5):
                return False
        return True

    def _check_zip_integrity(self) -> bool:
        return check_integrity(self.zip_file, self.zip_md5)

    def download(self) -> None:
        if self._check_zip_integrity():
            print("Archive already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.zip_md5)

    def extra_repr(self) -> str:
        return f"Split: {self.split.capitalize()}"

    def _parse_image(self, path) -> np.ndarray:
        img = Image.open(path)
        np_img = np.array(img)
        assert np_img.ndim in (2, 3), f'Image dim shoud be 2 or 3, but is {np_img.ndim}'
        assert np_img.shape[:2] == (self.INPUT_SIZE,) * 2, f'Illegal shape of {np_img.shape}'
        if np_img.ndim == 2:
            np_img = np.stack((np_img,) * 3, axis=-1)
        return np_img

    def _parse_dataset(self):
        '''
        generates npy files from the original folder dataset
        '''
        print(f'Parsing {self.split} data...')
        samples = []
        iter = self._classes if self.train else range(1)
        for cls in tqdm(iter, desc=" outer", position=0):
            if self.train:
                boxes_file = self.split_dir / cls / (cls + '_boxes.txt')
            else:
                boxes_file = self.split_dir / 'val_annotations.txt'
            with boxes_file.open() as boxes_file:
                lines = boxes_file.readlines()

            for line in tqdm(lines, position=1, leave=False):
                if self.train:
                    filename, *bbox = line.rstrip().split('\t')
                    path = self.split_dir / cls / 'images' / filename
                else:
                    filename, cls, *bbox = line.rstrip().split('\t')
                    path = self.split_dir / 'images' / filename

                bbox = map(int, bbox)
                image = self._parse_image(path)
                target = self.class_to_idx[cls]
                samples.append((image, target, bbox))

        image, target, bboxes = zip(*samples)
        self.data = np.stack(image)
        self.targets = np.array(target)
        self.bboxes = np.array(bboxes)

