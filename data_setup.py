import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Custom wrapper to shift MNIST labels (so no overlap with CIFAR 0-9)
class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label + 10  # Shift to 10-19

def get_hetero_dataloaders(batch_size=32):
    """
    Sets up train and test data loaders for 3 clients:
    - Client 1: CIFAR-10 vehicles (labels 0,1,8,9)
    - Client 2: CIFAR-10 animals (labels 2,3,4,5,6,7)
    - Client 3: MNIST digits (shifted to 10-19)
    All images: 32x32x3 channels.
    Returns train and test loaders separately.
    """
    print("⬇️ Setting up Heterogeneous Data...")

    # Transforms for CIFAR
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Transforms for MNIST (grayscale to 3-channel)
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load train datasets
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)

    # Load test datasets
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

    # Client 1 train: Vehicles
    c1_train_idx = [i for i, label in enumerate(cifar_train.targets) if label in [0, 1, 8, 9]]
    client1_train = Subset(cifar_train, c1_train_idx)

    # Client 2 train: Animals
    c2_train_idx = [i for i, label in enumerate(cifar_train.targets) if label in [2, 3, 4, 5, 6, 7]]
    client2_train = Subset(cifar_train, c2_train_idx)

    # Client 3 train: MNIST remapped
    client3_train = RemappedDataset(mnist_train)

    # Similar for test sets (smaller subsets for speed)
    c1_test_idx = [i for i, label in enumerate(cifar_test.targets) if label in [0, 1, 8, 9]]
    client1_test = Subset(cifar_test, c1_test_idx)

    c2_test_idx = [i for i, label in enumerate(cifar_test.targets) if label in [2, 3, 4, 5, 6, 7]]
    client2_test = Subset(cifar_test, c2_test_idx)

    client3_test = RemappedDataset(mnist_test)

    # DataLoaders (batches for training)
    train_l1 = DataLoader(client1_train, batch_size=batch_size, shuffle=True)
    train_l2 = DataLoader(client2_train, batch_size=batch_size, shuffle=True)
    train_l3 = DataLoader(client3_train, batch_size=batch_size, shuffle=True)

    test_l1 = DataLoader(client1_test, batch_size=batch_size, shuffle=False)
    test_l2 = DataLoader(client2_test, batch_size=batch_size, shuffle=False)
    test_l3 = DataLoader(client3_test, batch_size=batch_size, shuffle=False)

    print(f"✅ Client 1 Train/Test: {len(client1_train)}/{len(client1_test)} items")
    print(f"✅ Client 2 Train/Test: {len(client2_train)}/{len(client2_test)} items")
    print(f"✅ Client 3 Train/Test: {len(client3_train)}/{len(client3_test)} items")

    return (train_l1, train_l2, train_l3), (test_l1, test_l2, test_l3)