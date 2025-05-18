
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

class XrayThreatDataset(Dataset):

    
    def __init__(self, image_dir, annotation_dir, class_map, transforms=None, split='train'):
      
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.class_map = class_map
        self.transforms = transforms
        self.split = split
        
        # Load all annotation files
        self.annotations = []
        self._load_annotations()
        
    def _load_annotations(self):
        
        annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.txt')]
        
        for annotation_file in annotation_files:
            file_path = os.path.join(self.annotation_dir, annotation_file)
            
            with open(file_path, 'r') as f:
                content = f.read().strip()
                
                # have to check datset not good some empty 
                if not content:
                    continue
                
                # Parse annotation
                parts = content.split()
                
                # Ensure the annotation has the correct format
                if len(parts) < 6:
                    print(f"Warning: Invalid annotation format in {annotation_file}. Skipping.")
                    continue
                
                image_filename = parts[0]
                class_name = parts[1]
                
                # Skip if class is not in class map
                if class_name not in self.class_map:
                    print(f"Warning: Unknown class {class_name} in {annotation_file}. Skipping.")
                    continue
                
                # Parse bounding box coordinates
                try:
                    x1 = int(parts[2])
                    y1 = int(parts[3])
                    x2 = int(parts[4])
                    y2 = int(parts[5])
                except ValueError:
                    print(f"Warning: Invalid bounding box coordinates in {annotation_file}. Skipping.")
                    continue
                
                # Check if image file exists
                image_path = os.path.join(self.image_dir, image_filename)
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_filename} not found. Skipping.")
                    continue
                
                # Add annotation to list
                self.annotations.append({
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'class_name': class_name,
                    'class_idx': self.class_map[class_name],
                    'bbox': [x1, y1, x2, y2]
                })
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
      
        annotation = self.annotations[idx]
        
        # Load image
        image = Image.open(annotation['image_path']).convert('RGB')
        
        # Create target dictionary
        target = {
            'boxes': torch.tensor([annotation['bbox']], dtype=torch.float32),
            'labels': torch.tensor([annotation['class_idx']], dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor([(annotation['bbox'][2] - annotation['bbox'][0]) * 
                                 (annotation['bbox'][3] - annotation['bbox'][1])]),
            'iscrowd': torch.tensor([0])
        }
        
        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return {
            'image': image,
            'target': target,
            'metadata': {
                'image_filename': annotation['image_filename'],
                'class_name': annotation['class_name'],
                'image_path': annotation['image_path']
            }
        }


class XrayThreatDatasetMultiBox(Dataset):
  
    def __init__(self, image_dir, annotation_dir, class_map, transforms=None, split='train'):
        
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.class_map = class_map
        self.transforms = transforms
        self.split = split
        
        # Dictionary to store annotations by image
        self.image_annotations = {}
        self._load_annotations()
        
        # Convert to list for indexing
        self.images = list(self.image_annotations.keys())
        
    def _load_annotations(self):
        
        annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.txt')]
        
        for annotation_file in annotation_files:
            file_path = os.path.join(self.annotation_dir, annotation_file)
            
            with open(file_path, 'r') as f:
                content = f.read().strip()
                
                # Skip empty files
                if not content:
                    continue
                
                # Parse annotation
                parts = content.split()
                
                # Ensure the annotation has the correct format
                if len(parts) < 6:
                    print(f"Warning: Invalid annotation format in {annotation_file}. Skipping.")
                    continue
                
                image_filename = parts[0]
                class_name = parts[1]
                
                # Skip if class is not in class map
                if class_name not in self.class_map:
                    print(f"Warning: Unknown class {class_name} in {annotation_file}. Skipping.")
                    continue
                
                # Parse bounding box coordinates
                try:
                    x1 = int(parts[2])
                    y1 = int(parts[3])
                    x2 = int(parts[4])
                    y2 = int(parts[5])
                except ValueError:
                    print(f"Warning: Invalid bounding box coordinates in {annotation_file}. Skipping.")
                    continue
                
                # Check if image file exists
                image_path = os.path.join(self.image_dir, image_filename)
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_filename} not found. Skipping.")
                    continue
                
                # Add annotation to dictionary, grouped by image
                if image_filename not in self.image_annotations:
                    self.image_annotations[image_filename] = {
                        'image_path': image_path,
                        'boxes': [],
                        'class_names': [],
                        'class_indices': []
                    }
                
                self.image_annotations[image_filename]['boxes'].append([x1, y1, x2, y2])
                self.image_annotations[image_filename]['class_names'].append(class_name)
                self.image_annotations[image_filename]['class_indices'].append(self.class_map[class_name])
    
    def __len__(self):

        return len(self.images)# debug
    
    def __getitem__(self, idx):
      
        image_filename = self.images[idx]
        annotation = self.image_annotations[image_filename]
        
        # Load image
        image = Image.open(annotation['image_path']).convert('RGB')
        
        # Create target dictionary
        boxes = torch.tensor(annotation['boxes'], dtype=torch.float32)
        labels = torch.tensor(annotation['class_indices'], dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': torch.tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in annotation['boxes']]),
            'iscrowd': torch.zeros((len(annotation['boxes']),), dtype=torch.int64)
        }
        
        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return {
            'image': image,
            'target': target,
            'metadata': {
                'image_filename': image_filename,
                'class_names': annotation['class_names'],
                'image_path': annotation['image_path']
            }
        }


def create_data_loaders(config):

    from src.data.transforms import get_transforms
    
    # Create class map
    class_map = {class_name: idx for idx, class_name in enumerate(config.CLASSES) if class_name != 'background'}
    
    # Create transforms
    transforms = get_transforms(config)
    
    # Create dataset
    dataset = XrayThreatDatasetMultiBox(
        image_dir=config.TRAIN_IMAGES_DIR,
        annotation_dir=config.TRAIN_ANNOTATIONS_DIR,
        class_map=class_map,
        transforms=transforms['train'],
        split='train'
    )
    
    # Split dataset into train, validation, and test sets
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=1 - config.TRAIN_VAL_SPLIT,
        random_state=config.RANDOM_SEED
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'class_map': class_map
    }


def collate_fn(batch):
  
    images = []
    targets = []
    metadata = []
    
    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
        metadata.append(item['metadata'])
    
    return images, targets, metadata
