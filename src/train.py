import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.norm = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(self.norm(x))
        x = torch.flatten(self.pool(x), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_eval(model, train_loader, test_loader):
    # Set up the model and optimizer
    num_epochs = 50
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())
    # Run one or more epochs
    for epoch in range(num_epochs):
        print(f"---- Beginning epoch {epoch} ----")
        model.train()
        progress_bar = tqdm(train_loader)
        # Train on an epoch of minibatches
        for X, y in progress_bar:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            loss = F.cross_entropy(y_hat, y)
            progress_bar.set_postfix_str(f"train loss: {loss.item():.4f}")
            loss.backward()
            opt.step()
            opt.zero_grad()
        # Evaluate the model at the end of the epoch
        model.eval()
        num_right = 0
        eval_size = 0
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            num_right += (y_hat.argmax(dim=1) == y).sum().item()
            eval_size += len(y)
        acc_percent = 100 * num_right / eval_size
        print(f"Epoch {epoch} validation accuracy: {acc_percent:.2f}%")

def main():
    datadir = './data'
    batch_size = 1024

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root=datadir, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=datadir, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    model = Net()
    train_and_eval(model, trainloader, testloader)

if __name__ == "__main__":
    main()