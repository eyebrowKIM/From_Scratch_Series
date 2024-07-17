import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

from train import UNet

def test_unet(val_data, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    iou_scores = []
    
    model = UNet(n_classes=1000).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    val_data, val_label = next(iter(val_data))
    val_data = val_data.to(device)
    val_label = val_label.squeeze(1)
    
    inverse_transform = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225)),
    ])
    
    test_batch_size = 8
    
    fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5))
    
    for i in range(test_batch_size):
        val_data_sample = val_data[i:i+1]  # Get the i-th sample
        val_label_sample = val_label[i]  # Get the i-th label

        # Transform val_data_sample if needed
        val_data_sample = inverse_transform(val_data_sample)
        
        output = model(val_data_sample)
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        
        output = output.cpu().numpy()
        val_label_sample = val_label_sample.cpu().numpy()
        
        # IOU score
        intersection = np.logical_and(output, val_label_sample)
        union = np.logical_or(output, val_label_sample)
        iou_score = np.sum(intersection) / np.sum(union)
        
        iou_scores.append(iou_score)
        
        # Plotting
        axes[i, 0].imshow(val_data_sample.cpu().numpy().squeeze().transpose(1, 2, 0))
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(val_label_sample.squeeze(), cmap='gray')
        axes[i, 1].set_title("Label")
        axes[i, 2].imshow(output.squeeze(), cmap='gray')
        axes[i, 2].set_title(f"Output - IOU: {iou_score:.4f}")

    plt.tight_layout()
    plt.show()
    
    accuracy = sum(iou_scores) / len(iou_scores)
    
    print(f'Accuracy: {accuracy:.4f}')