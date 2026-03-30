import os
import torch
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class LiverDataset(Dataset):
    def __init__(self,root_dir,transform=None,target_transform =None,image_size=(512,512)):
        # get root dir path
        images = load_images(root_dir)
        self.images = images
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size

    # index：image name ,class_number:mask class number 
    def __getitem__(self, index):
        train_dir,mask_dir = self.images[index]
        train = Image.open(train_dir)
        mask = Image.open(mask_dir)
        # sample trans tensor
        if self.transform is not None:
            train = self.transform(train)
        # mask trans tensor
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        else:
            if self.image_size is not None:
                # resize mask image  
                mask = mask.resize(self.image_size)
            mask = np.array(mask)
            mask = torch.from_numpy(mask)
        return train,mask
    
    # get train image number
    def __len__(self):
        return len(self.images)
    
def load_images(root_dir):
    images=[]
    root_name = os.listdir(root_dir)
    for name in root_name:
        image = os.path.join(root_dir,name)
        mask = image.replace('roi','class_a')
        images.append((image,mask))
    return images

if __name__ == '__main__':
    train_path = r'D:\AI_code\python\dataset\train_images\roi'
    weight = 256
    height = 128
    x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((weight, height)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
    liver_dataset = LiverDataset(train_path, 
                                 transform=x_transforms, target_transform=None, image_size=(weight, height))

