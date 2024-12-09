import os
import zipfile
import numpy as np
import random
import torch
from itertools import chain

import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms as T
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10, CIFAR100, VOCSegmentation
from tqdm import tqdm


class CIFARData(pl.LightningDataModule):
    def __init__(self, nr_classes):
        super().__init__()
#        self.hparams = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.nr_classes = nr_classes

    def download_weights():
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )

        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)

        # Total size in Mebibyte
        total_size = int(r.headers.get("content-length", 0))
        block_size = 2 ** 20  # Mebibyte
        t = tqdm(total=total_size, unit="MiB", unit_scale=True)

        with open("state_dicts.zip", "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("Error, something went wrong")

        print("Download successful. Unzipping file...")
        path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
        directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
            print("Unzip file successful!")

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        random.seed(12345)
        torch.manual_seed(12345)
        np.random.seed(12345)
        transform = T.Compose(
            [
                # v2.ColorJitter(brightness=(0.1,2)),
                # v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.5)),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        
        if self.nr_classes == 10:
            dataset = CIFAR10(root="/home/nadia_dobreva/PyTorch_CIFAR10/data/cifar10/", train=False, transform=transform, download=True)
        elif self.nr_classes == 100:
            dataset = CIFAR100(root="/home/nadia_dobreva/PyTorch_CIFAR10/data/cifar100/", train=False, transform=transform, download=True)
        else:
            print("invalid dataset requested")
            exit(0)
            
        subset_size = 50
        rand_start = 7037 #np.random.randint(0,dataset.__len__())
        small_dataset = Subset(dataset, range(rand_start, rand_start + subset_size))
        val_dataloader = DataLoader(
            small_dataset,
            batch_size=subset_size-20,
            num_workers=4,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )

        rest_of_dataset = Subset(dataset, list(chain(range(0, rand_start), range(rand_start+subset_size+1, len(dataset)))))
        test_dataloader = DataLoader(
            rest_of_dataset,
            batch_size=128,
            num_workers=4,
            drop_last=True,
            pin_memory=True,
        )

        return val_dataloader, test_dataloader

    def test_dataloader(self):
        return self.val_dataloader()



def load_PascalVOC():
    random.seed(12345)
    torch.manual_seed(12345)
    np.random.seed(12345)

    transform = T.Compose(
            [
                v2.Resize(size=(520,520)),
                v2.ToImage(),
                v2.ToDtype(torch.float, scale=True),
                v2.PILToTensor(),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    target_transform = T.Compose([v2.Resize(size=(520,520)), v2.PILToTensor(),])
    dataset = VOCSegmentation(root="/home/nadia_dobreva/PyTorch_CIFAR10/data/pascal/", image_set='val', transform=transform, target_transform=target_transform)

    subset_size = 50
    rand_start = 120 #np.random.randint(0,dataset.__len__())
    small_dataset = Subset(dataset, range(rand_start, rand_start + subset_size))
    val_dataloader = DataLoader(
        small_dataset,
        batch_size=subset_size-20,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
    )

    rest_of_dataset = Subset(dataset, list(chain(range(0, rand_start), range(rand_start+subset_size+1, len(dataset)))))
    test_dataloader = DataLoader(
        rest_of_dataset,
        batch_size=128,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    return val_dataloader, test_dataloader
