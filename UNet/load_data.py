import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_sample(data_dir = './dataset', from_='train'):
    
    if from_ == 'train':
        train_fns = sorted(os.listdir(os.path.join(data_dir, 'train')))
        train_dir = os.path.join(data_dir, 'train')
        
        data_path = os.path.join(train_dir, train_fns[0])
        sample_image = Image.open(data_path)
    
    elif from_ == 'val':
        test_fns = sorted(os.listdir(os.path.join(data_dir, 'test')))
        test_dir = os.path.join(data_dir, 'test')
        
        data_path = os.path.join(test_dir, test_fns[0])
        sample_image = Image.open(data_path)
    
    else:
        raise ValueError('from_ must be either "train" or "val"')
    
    plt.imshow(sample_image)
    plt.show()
    

class CityscapeDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_fns = os.listdir(image_dir)

        color_array = np.random.choice(256, size=(1000, 3))
        self.label_model = KMeans(n_clusters=1000, random_state=0).fit(color_array)
        
    def __len__(self) :
        return len(self.image_fns)
        
    def __getitem__(self, index) :
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp)
        image = np.array(image)
        cityscape, label = self.split_image(image)
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
        label_class = torch.Tensor(label_class).long().unsqueeze(0)
        cityscape = self.transform(cityscape)
        return cityscape, label_class
        
    def split_image(self, image) :
        image = np.array(image)
        cityscape, label = image[ : , :256, : ], image[ : , 256: , : ]
        return cityscape, label
        
    def transform(self, image) :
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.56, 0.406), std = (0.229, 0.224, 0.225))
        ])
        return transform_ops(image)          

def dataload(batch_size=16):
    train_dataset = CityscapeDataset('./dataset/train')
    val_dataset = CityscapeDataset('./dataset/val')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader


def main():
    get_sample()
    
    train_dataset = CityscapeDataset('./dataset/train')
    print(len(train_dataset))
    
    cityscape, label = train_dataset[0]
    print(cityscape.shape, label.shape)
    


if __name__ == '__main__':
    main()