import json
import os
import pickle
import random
import subprocess
import zipfile
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import List, Type, Dict

import requests
import numpy as np
# import spacy as spacy
import torchvision
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset
import torch
# from torchtext.vocab import Vocab


class BaseDataset(Dataset):
    def __init__(self) -> None:
        self.classes: List = None
        self.data: torch.Tensor = None
        self.targets: torch.Tensor = None
        self.train_data_transform = None
        self.train_target_transform = None
        self.general_data_transform = None
        self.general_target_transform = None
        self.enable_train_transform = True

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        if self.enable_train_transform and self.train_data_transform is not None:
            data = self.train_data_transform(data)
        if self.enable_train_transform and self.train_target_transform is not None:
            targets = self.train_target_transform(targets)
        if self.general_data_transform is not None:
            data = self.general_data_transform(data)
        if self.general_target_transform is not None:
            targets = self.general_target_transform(targets)
        return data, targets

    def __len__(self):
        return len(self.targets)


class FEMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).float().reshape(-1, 1, 28, 28)
        self.targets = torch.from_numpy(targets).long()
        self.classes = list(range(62))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class Synthetic(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).long()
        self.classes = list(range(len(self.targets.unique())))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CelebA(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).permute([0, -1, 1, 2]).float()
        self.targets = torch.from_numpy(targets).long()
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform
        self.classes = [0, 1]


class MedMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        self.classes = list(range(11))
        self.data = (
            torch.Tensor(np.load(root / "raw" / "xdata.npy")).float().unsqueeze(1)
        )
        self.targets = (
            torch.Tensor(np.load(root / "raw" / "ydata.npy")).long().squeeze()
        )
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class COVID19(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        self.data = (
            torch.Tensor(np.load(root / "raw" / "xdata.npy"))
            .permute([0, -1, 1, 2])
            .float()
        )
        self.targets = (
            torch.Tensor(np.load(root / "raw" / "ydata.npy")).long().squeeze()
        )
        self.classes = [0, 1, 2, 3]
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class USPS(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.USPS(root / "raw", True, download=True)
        test_part = torchvision.datasets.USPS(root / "raw", False, download=True)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long()
        test_targets = torch.Tensor(test_part.targets).long()

        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = list(range(10))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class SVHN(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.SVHN(root / "raw", "train", download=True)
        test_part = torchvision.datasets.SVHN(root / "raw", "test", download=True)
        train_data = torch.Tensor(train_part.data).float()
        test_data = torch.Tensor(test_part.data).float()
        train_targets = torch.Tensor(train_part.labels).long()
        test_targets = torch.Tensor(test_part.labels).long()

        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = list(range(10))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class MNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.MNIST(root, True, download=True)
        test_part = torchvision.datasets.MNIST(root, False)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class FashionMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.FashionMNIST(root, True, download=True)
        test_part = torchvision.datasets.FashionMNIST(root, False, download=True)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class EMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        split = None
        if isinstance(args, Namespace):
            split = args.emnist_split
        elif isinstance(args, dict):
            split = args["emnist_split"]
        train_part = torchvision.datasets.EMNIST(
            root, split=split, train=True, download=True
        )
        test_part = torchvision.datasets.EMNIST(
            root, split=split, train=False, download=True
        )
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CIFAR10(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.CIFAR10(root, True, download=True)
        test_part = torchvision.datasets.CIFAR10(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CIFAR100(BaseDataset):
    def __init__(
        self,
        root,
        args,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.CIFAR100(root, True, download=True)
        test_part = torchvision.datasets.CIFAR100(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform
        super_class = None
        if isinstance(args, Namespace):
            super_class = args.super_class
        elif isinstance(args, dict):
            super_class = args["super_class"]

        if super_class:
            # super_class: [sub_classes]
            CIFAR100_SUPER_CLASS = {
                0: ["beaver", "dolphin", "otter", "seal", "whale"],
                1: ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                2: ["orchid", "poppy", "rose", "sunflower", "tulip"],
                3: ["bottle", "bowl", "can", "cup", "plate"],
                4: ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                5: ["clock", "keyboard", "lamp", "telephone", "television"],
                6: ["bed", "chair", "couch", "table", "wardrobe"],
                7: ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                8: ["bear", "leopard", "lion", "tiger", "wolf"],
                9: ["cloud", "forest", "mountain", "plain", "sea"],
                10: ["bridge", "castle", "house", "road", "skyscraper"],
                11: ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                12: ["fox", "porcupine", "possum", "raccoon", "skunk"],
                13: ["crab", "lobster", "snail", "spider", "worm"],
                14: ["baby", "boy", "girl", "man", "woman"],
                15: ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                16: ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                17: ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                18: ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                19: ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
            }
            mapping = {}
            for super_cls, sub_cls in CIFAR100_SUPER_CLASS.items():
                for cls in sub_cls:
                    mapping[cls] = super_cls
            new_targets = []
            for cls in self.targets:
                new_targets.append(mapping[self.classes[cls]])
            self.targets = torch.tensor(new_targets, dtype=torch.long)


class TinyImagenet(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        print(f"path:{root}/raw")
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/tiny_imagenet.sh` to download the dataset first."
            )
        self.classes = pd.read_table(
            root / "raw/wnids.txt", sep="\t", engine="python", header=None
        )[0].tolist()

        if not os.path.isfile(root / "data.pt") or not os.path.isfile(
            root / "targets.pt"
        ):
            mapping = dict(zip(self.classes, list(range(len(self.classes)))))
            data = []
            targets = []
            for cls in os.listdir(root / "raw" / "train"):
                for img_name in os.listdir(root / "raw" / "train" / cls / "images"):
                    img = pil_to_tensor(
                        Image.open(root / "raw" / "train" / cls / "images" / img_name)
                    ).float()
                    if img.shape[0] == 1:
                        img = torch.expand_copy(img, [3, 64, 64])
                    data.append(img)
                    targets.append(mapping[cls])

            table = pd.read_table(
                root / "raw/val/val_annotations.txt",
                sep="\t",
                engine="python",
                header=None,
            )
            test_classes = dict(zip(table[0].tolist(), table[1].tolist()))
            for img_name in os.listdir(root / "raw" / "val" / "images"):
                img = pil_to_tensor(
                    Image.open(root / "raw" / "val" / "images" / img_name)
                ).float()
                if img.shape[0] == 1:
                    img = torch.expand_copy(img, [3, 64, 64])
                data.append(img)
                targets.append(mapping[test_classes[img_name]])
            torch.save(torch.stack(data), root / "data.pt")
            torch.save(torch.tensor(targets, dtype=torch.long), root / "targets.pt")

        self.data = torch.load(root / "data.pt")
        self.targets = torch.load(root / "targets.pt")
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CINIC10(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        print(f"path:{root}/raw")
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/cinic10.sh` to download the dataset first."
            )
        self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        if not os.path.isfile(root / "data.pt") or not os.path.isfile(
            root / "targets.pt"
        ):
            data = []
            targets = []
            mapping = dict(zip(self.classes, range(10)))
            for folder in ["test", "train", "valid"]:
                for cls in os.listdir(Path(root) / "raw" / folder):
                    for img_name in os.listdir(root / "raw" / folder / cls):
                        img = pil_to_tensor(
                            Image.open(root / "raw" / folder / cls / img_name)
                        ).float()
                        if img.shape[0] == 1:
                            img = torch.expand_copy(img, [3, 32, 32])
                        data.append(img)
                        targets.append(mapping[cls])
            torch.save(torch.stack(data), root / "data.pt")
            torch.save(torch.tensor(targets, dtype=torch.long), root / "targets.pt")

        self.data = torch.load(root / "data.pt")
        self.targets = torch.load(root / "targets.pt")
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class DomainNet(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/domain.sh` to download the dataset first."
            )
        targets_path = root / "targets.pt"
        metadata_path = root / "metadata.json"
        filename_list_path = root / "filename_list.pkl"
        if not (
            os.path.isfile(targets_path)
            and os.path.isfile(metadata_path)
            and os.path.isfile(filename_list_path)
        ):
            raise RuntimeError(
                "Run data/domain/preprocess.py to preprocess DomainNet first."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        with open(filename_list_path, "rb") as f:
            self.filename_list = pickle.load(f)

        self.classes = list(metadata["classes"].keys())
        self.targets = torch.load(targets_path)
        self.pre_transform = transforms.Compose(
            [
                transforms.Resize([metadata["image_size"], metadata["image_size"]]),
                transforms.ToTensor(),
            ]
        )
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform

    def __getitem__(self, index):
        data = self.pre_transform(Image.open(self.filename_list[index]).convert("RGB"))
        targets = self.targets[index]
        if self.enable_train_transform and self.train_data_transform is not None:
            data = self.train_data_transform(data)
        if self.enable_train_transform and self.train_target_transform is not None:
            targets = self.train_target_transform(targets)
        if self.general_data_transform is not None:
            data = self.general_data_transform(data)
        if self.general_target_transform is not None:
            targets = self.general_target_transform(targets)
        return data, targets



def download_csv(url, local_filename):
    """
    Download a CSV from a URL and save it to a local file.

    Parameters:
    - url (str): The URL of the CSV file.
    - local_filename (str): The local path where the CSV should be saved.
    """
    response = requests.get(url)

    # Ensure the request was successful.
    response.raise_for_status()

    with open(local_filename, 'wb') as f:
        f.write(response.content)


class IMDB(BaseDataset):
    def __init__(
            self,
            root,
            args=None,
            general_data_transform=None,
            general_target_transform=None,
            train_data_transform=None,
            train_target_transform=None,
    ):
        super().__init__()
        fname='imdb.csv'
        if not os.path.isfile(fname):
            url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
            local_filename = fname
            download_csv(url, local_filename)

        self.dataframe = pd.read_csv(fname)
        self.data = self.dataframe[self.dataframe.columns[0]].values
        labels = self.dataframe[self.dataframe.columns[1]].values
        self.targets = list(map(lambda x: 1 if x == 'positive' else 0, labels))
        # spacy_en = spacy.load('en_core_web_sm')

        # Tokenization function
        # def tokenize_en(text):
        #     return [tok.text for tok in spacy_en.tokenizer(text)]

        # tokenized_texts = self.dataframe['text'].apply(tokenize_en)
        # word_counter = Counter([word for sentence in tokenized_texts for word in sentence])
        # vocab = Vocab(word_counter, min_freq=1, specials=('<unk>', '<pad>', '<bos>', '<eos>'))

        self.classes = [
            "positive",
            "negative"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # row = self.data.iloc[idx].to_dict()
        return self.data[idx], self.targets[idx]


class SENT140(BaseDataset):
    def __init__(
            self,
            root,
            args=None,
            general_data_transform=None,
            general_target_transform=None,
            train_data_transform=None,
            train_target_transform=None,
    ):
        super().__init__()
        fname = 'training.csv'
        if not os.path.isfile(fname):
            url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
            local_filename = "trainingandtestdata.zip"
            self.download_and_extract(url, local_filename)

        self.dataframe = pd.read_csv(fname, encoding="ISO-8859-1", header=None)
        # Column 0 contains labels, and Column 5 contains the text in Sent140 dataset
        self.data = self.dataframe[5].values
        # Assuming labels are 0 (negative), 2 (neutral), and 4 (positive)
        labels = self.dataframe[0].values
        self.targets = list(map(lambda x: 1 if x == 4 else 0, labels))

        self.classes = [
            "negative",
            "positive"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def download_and_extract(self, url, local_filename):
        # Check if the file exists, if not, download it using wget
        if not os.path.exists("trainingandtestdata.zip"):
            url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
            subprocess.run(["wget", "--no-check-certificate", url])

        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall('.')

        # Rename the files
        os.rename("training.1600000.processed.noemoticon.csv", "training.csv")
        os.rename("testdata.manual.2009.06.14.csv", "test.csv")

        # Remove the zip file
        os.remove("trainingandtestdata.zip")



DATASETS: Dict[str, Type[BaseDataset]] = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "mnist": MNIST,
    "emnist": EMNIST,
    "fmnist": FashionMNIST,
    "femnist": FEMNIST,
    "medmnistS": MedMNIST,
    "medmnistC": MedMNIST,
    "medmnistA": MedMNIST,
    "covid19": COVID19,
    "celeba": CelebA,
    "synthetic": Synthetic,
    "svhn": SVHN,
    "usps": USPS,
    "tiny_imagenet": TinyImagenet,
    "cinic10": CINIC10,
    "domain": DomainNet,
    "imdb": IMDB,
    "sent140": SENT140
}
