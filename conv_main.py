import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from conv import Conv

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to("cuda"), y.to("cuda")
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cuda"), y.to("cuda")
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {100*correct:0.1f}%, Avg loss: {test_loss:>8f}\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using {device}")

    training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
            )

    test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
            )

    train_dataloader = DataLoader(training_data, batch_size=64, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=64, pin_memory=True)

    model = Conv().to(device)

    learning_rate = 1e-3
    batch_size = 64
    epochs = 40

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        print(f"Epoch {i+1}\n")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    ran_num = np.random.randint(1, 100)
    torch.save(model.state_dict(), "model_{ran_num}".format(ran_num=ran_num))
    print("done")

