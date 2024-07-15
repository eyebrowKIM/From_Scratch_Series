from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets import load_dataset


class ImageNetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]['image'], self.dataset[idx]['label']


def data_ImageNet():
    train_data = load_dataset('ILSVRC/imagenet-1k', cache_dir='./dataset', split='train')
    eval_data = load_dataset('ILSVRC/imagenet-1k', cache_dir='./dataset', split='validation')
    
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )
    
    train_data = ImageNetDataset(train_data)
    eval_data = ImageNetDataset(eval_data)
    
    train_data = DataLoader(train_data, batch_size=100, shuffle=True)
    eval_data = DataLoader(eval_data, batch_size=100, shuffle=False)
    
    return train_data, eval_data

data_ImageNet()