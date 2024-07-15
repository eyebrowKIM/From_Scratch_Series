from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def data_MNIST(batch_size=100):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )

    train_loader = datasets.MNIST(root='dataset/',
                                    train = True,
                                    download = True,
                                    transform = transform
                                    )

    test_loader = datasets.MNIST(root='dataset/',
                                    train = False,
                                    download = True,
                                    transform = transform
                                    )

    train_data = DataLoader(dataset=train_loader, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(dataset=test_loader, batch_size=batch_size, shuffle=True)

    return train_data, test_data