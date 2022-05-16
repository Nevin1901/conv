import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import Net

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
            )

    print(test_data[0][1])
    # print(torch.flatten(test_data[0][0]))
    image = test_data[0][0].to(device)
    print(image.shape)

    model = Net().to(device)
    model.load_state_dict(torch.load("model.pth"))

    with torch.no_grad():
        prediction = model(image).to("cpu")
        pred_class = np.argmax(prediction)

        # for matplotlib, since 1, 28, 28 -> 28, 28, 1
        out = image.reshape(28, 28, 1).to("cpu")
        plt.imshow(out)
        plt.title(f"prediction: {pred_class} - actual {test_data[0][1]}")
        plt.show()


