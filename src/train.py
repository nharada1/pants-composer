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

from streaming import StreamingDataset

class ResNet18(ComposerModel):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(num_classes=63)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def forward(self, batch):
        inputs, _ = batch
        inputs = inputs[:, 0:3, :, :]
        return self.model(inputs)

    def loss(self, outputs, batch):
        _, targets = batch
        oh = F.one_hot(targets, num_classes=63)
        oh = oh.squeeze().float()
        return F.cross_entropy(outputs, oh)


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


def main(args):
    batch_size = 32
    mean=np.array(
        [349.23, 339.76, 378.58, 418.42, 275.86, 431.82, 495.65, 435.05]
    )
    std=np.array(
        [78.67, 105.54, 142.05, 177.00, 132.29, 151.65, 194.00, 166.27]
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((256, 256)), transforms.Normalize(mean, std)]
    )
    local_dir = args.local
    remote_dir = None

    trainset = CustomDataset(local=local_dir, remote=remote_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, num_workers=8
    )

    model = ResNet18()
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
    parser.add_argument('--local', type=str, help='The path for the local directory.')
    args = parser.parse_args()

    main(args)
