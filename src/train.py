from typing import Any

import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from composer.models import ComposerModel
from composer import Trainer
from composer.utils import dist

from moonshine.models.unet import UNet as MSUNet

from streaming import StreamingDataset # pants: no-infer-dep

# Datasets
class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, size, image_size, num_classes, transform=None):
        self.size = size
        self.shape = image_size
        self.n_classes = num_classes
        self.transform = transform

    def __getitem__(self, index):
        x = np.random.randint(low=0, high=3000, size=self.shape).astype(np.float16)
        y = np.random.randint(low=0, high=self.n_classes, size=(1,)).astype(np.int64)

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.size

class CustomDataset(StreamingDataset):
    def __init__(self, local, remote, transform):
        super().__init__(local=local, remote=remote)
        self.transform = transform

    def __getitem__(self, idx: int) -> Any:
        obj = super().__getitem__(idx)
        image, label = obj["image"], obj["label"]

        if self.transform:
            image = self.transform(image.astype(np.float32))

        return image, label


# Models
class UNetModule(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = MSUNet(name="unet50_fmow_full")
        self.classifier = torch.nn.Conv2d(32, self.n_classes, (1, 1))

    def forward(self, batch):
        x = self.backbone(batch)
        x = self.classifier(x)
        x = x.mean((2, 3))
        return x


class UNet(ComposerModel):
    def __init__(self):
        super().__init__()
        self.n_classes = 62
        self.model = UNetModule(n_classes=self.n_classes)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def forward(self, batch):
        return self.model(batch[0])

    def loss(self, outputs, batch):
        _, targets = batch
        oh = F.one_hot(targets, num_classes=self.n_classes)
        oh = oh.squeeze().float()
        return F.cross_entropy(outputs, oh)


def main(args):
    batch_size = 32
    mean = np.array([349.23, 339.76, 378.58, 418.42, 275.86, 431.82, 495.65, 435.05])
    std = np.array([78.67, 105.54, 142.05, 177.00, 132.29, 151.65, 194.00, 166.27])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.RandomCrop((256, 256)),
            transforms.Normalize(mean, std),
        ]
    )
    local_dir = args.local
    remote_dir = None

    if args.fake:
        trainset = FakeDataset(size=100_000, image_size=(8, 256, 256), num_classes=62)
        sampler = dist.get_sampler(trainset)
    else:
        trainset = CustomDataset(
            local=local_dir, remote=remote_dir, transform=transform
        )
        sampler = None

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        num_workers=8,
        sampler=sampler,
    )

    model = UNet()
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        optimizers=model.optim,
        max_duration="100000ba",
        device="gpu",
    )

    trainer.fit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=str, help="The path for the local directory.")
    parser.add_argument("--fake", action="store_true", help="Use fake data.")
    args = parser.parse_args()

    main(args)
