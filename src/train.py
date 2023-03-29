import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from composer.models import ComposerModel
from composer import Trainer
from composer.utils import dist


class ResNet18(ComposerModel):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def forward(self, batch):
        inputs, _ = batch
        return self.model(inputs)

    def loss(self, outputs, batch):
        _, targets = batch
        return F.cross_entropy(outputs, targets)


def main():
    datadir = "./data"
    batch_size = 1024

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=datadir, train=True, download=True, transform=transform
    )
    sampler = dist.get_sampler(trainset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        trainset, sampler=sampler, batch_size=batch_size, num_workers=2
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
    main()
