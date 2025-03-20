import torch
from torchvision import datasets, transforms
from typing import Tuple, List
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from PIL import ImageFilter
from tqdm import tqdm
import yaml

with open("variables.yaml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

CACHE = config['LOCAL_PATH']['CACHE']

transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :])
        ])

transform_train = transforms.Compose([
    transforms.Resize((256, 256)), # Resize
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)), # Zoom
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
    transforms.RandomRotation(15),  # Rotate
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Modify brightness, contrast, saturation and hue
    transforms.ToTensor(),  # Convert to tensor
    transforms.Lambda(lambda x: x[:3, :, :]),  # Remove alpha channel if it exists
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (recommanded for pre-trained model)
])

class CustomDataset(Dataset) :
    def __init__(self, data, transform=None) :
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        # Load image if imag_path is string else use the image
        if isinstance(img_path, str):
            img = Image.open(img_path)
        else:
            img = img_path
        
        if self.transform:
            img = self.transform(img)
        return img, label

def augment_dataset(dataset : Dataset, cache : str = CACHE) -> Dataset :
    """
    Augment the dataset
    
    :param dataset: the dataset to augment
    :param cache: the cache directory
    """
    print (f"Dataset de base : {len(dataset)} images")
    if not os.path.exists(cache):
        os.makedirs(cache)

    with tqdm(range(len(dataset)), desc="Augmentation") as t:
        for i in t:
            
            img_path, label = dataset[i]         
            name = os.path.basename(img_path)
            name, _ = name.split('.')
            
            img = Image.open(img_path)
            img = img.resize((256, 256))
            
            new_img = img.filter(ImageFilter.GaussianBlur(radius=5))  
            
            save_path = os.path.join(cache, f"{name}_blur.png")
            if not os.path.exists(save_path):
                new_img.save(save_path)
            dataset.append((save_path, label))

            save_path = os.path.join(cache, f"{name}_rotate.png")
            if not os.path.exists(save_path):
                new_img = img.rotate(45)
                new_img.save(save_path)
            dataset.append((save_path, label))
            
            save_path = os.path.join(cache, f"{name}_decrease_brightness.png")
            if not os.path.exists(save_path):
                new_img = img.point(lambda p: p * 0.5)
                new_img.save(save_path)
            dataset.append((save_path, label))
            
            save_path = os.path.join(cache, f"{name}_increase_brightness.png")
            if not os.path.exists(save_path):
                new_img = img.point(lambda p: p * 1.5)
                new_img.save(save_path)
            dataset.append((save_path, label))
            
            save_path = os.path.join(cache, f"{name}_increase_contrast.png")
            if not os.path.exists(save_path):
                new_img = img.convert("HSV")
                h, s, v = new_img.split()
                ns = s.point(lambda p: p * 1.5)
                new_img = Image.merge("HSV", (h, ns, v))
                new_img = new_img.convert("RGB")
                new_img.save(save_path)
            dataset.append((save_path, label))
            
            save_path = os.path.join(cache, f"{name}_decrease_contrast.png")
            if not os.path.exists(save_path):
                new_img = img.convert("HSV")
                h, s, v = new_img.split()
                ns = s.point(lambda p: p * 0.5)
                new_img = Image.merge("HSV", (h, ns, v))
                new_img = new_img.convert("RGB")
                dataset.append((new_img, label))
                new_img.save(save_path)
            dataset.append((save_path, label))
    
    print (f"Dataset augmenté : {len(dataset)} images")
    return dataset

def split_dataset(dataset : Dataset, test_size : float = 0.2) -> Tuple[Dataset, Dataset] :
    """
    Split the dataset into a training and a testing set
    
    :param dataset: the dataset to split
    :param test_size: the size of the testing set
    
    :return: tuple of the training and testing set    
    """
    data = dataset.imgs
    train_dataset, test_dataset = train_test_split(data, test_size=test_size, random_state=42)

    return train_dataset, test_dataset

def get_dataloader(data_dir : str = "classifier/data", batch_size : int = 32,
                   img_size : int = 256, num_workers : int = 0, transform_type : int = 0, 
                   augment_data : bool = True, test_size : float = 0.2) -> Tuple[DataLoader, DataLoader, List[str]] :
    """
    Get the dataloader for the dataset
    
    :param data_dir: the directory of the dataset
    :param batch_size: the batch size
    :param img_size: the size of the images
    :param num_workers: the number of workers
    :param transform_type: the type of transformation (0 : train, 1 : test)
    :param augment_data: whether to augment the data
    :param test_size: the size of the testing set
    
    :return: tuple of the training dataloader, testing dataloader and the classes
    """
    # Load basic dataset
    transform = transform_train if transform_type == 0 else transform_test
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    #print(f"Detecting classes : {dataset.classes}")

    if test_size == 0.0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), None, dataset.classes
    
    # Split dataset    
    train_dataset, test_dataset = split_dataset(dataset, test_size=test_size)

    # Data augmentation
    if augment_data:
        train_dataset = augment_dataset(train_dataset)

    train_dataset = CustomDataset(train_dataset, transform=dataset.transform)
    test_dataset = CustomDataset(test_dataset, transform=dataset.transform)
    
    # Create dataloaders
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return dataloader_train, dataloader_test, dataset.classes

def main() -> None:
    dataloader_train, dataloader_test, classes = get_dataloader()
    
    print(f"Detecting classes : {classes}")

    for images, labels in dataloader_train:
        print(f"Image batch : {images.shape}")
        print(f"Label batch : {labels}")
        break

if __name__ == "__main__":
    print("To train please got to ~/pfe/ and run \'python main.py dataloader\'")