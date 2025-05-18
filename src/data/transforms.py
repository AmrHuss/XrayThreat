

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np


class Compose:
   
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
  #idk check
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomHorizontalFlip:
   
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.height, image.width
            image = F.hflip(image)
            
            # Flip bounding boxes
            if "boxes" in target:
                boxes = target["boxes"]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
                
        return image, target


class RandomVerticalFlip:
    
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.height, image.width
            image = F.vflip(image)
            
            # Flip bounding boxes
            if "boxes" in target:
                boxes = target["boxes"]
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                target["boxes"] = boxes
                
        return image, target


class RandomRotation:
   
    def __init__(self, degrees):
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        
    def __call__(self, image, target):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        
        # Rotate image
        height, width = image.height, image.width
        center = (width / 2, height / 2)
        
        # Calculate rotation matrix
        angle_rad = angle * np.pi / 180
        alpha = np.cos(angle_rad)
        beta = np.sin(angle_rad)
        
        # Rotate image
        image = F.rotate(image, angle, expand=False)
        
        # Rotate bounding boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            
            # Convert boxes to corners
            corners = torch.zeros((boxes.shape[0], 4, 2), dtype=torch.float32)
            corners[:, 0, 0] = boxes[:, 0]  # x1
            corners[:, 0, 1] = boxes[:, 1]  # y1
            corners[:, 1, 0] = boxes[:, 2]  # x2
            corners[:, 1, 1] = boxes[:, 1]  # y1
            corners[:, 2, 0] = boxes[:, 2]  # x2
            corners[:, 2, 1] = boxes[:, 3]  # y2
            corners[:, 3, 0] = boxes[:, 0]  # x1
            corners[:, 3, 1] = boxes[:, 3]  # y2
            
            # Rotate corners
            corners = corners - torch.tensor([center[0], center[1]])
            rotated_corners = torch.zeros_like(corners)
            rotated_corners[:, :, 0] = corners[:, :, 0] * alpha - corners[:, :, 1] * beta
            rotated_corners[:, :, 1] = corners[:, :, 0] * beta + corners[:, :, 1] * alpha
            rotated_corners = rotated_corners + torch.tensor([center[0], center[1]])
            
            # Get new bounding boxes from rotated corners
            new_boxes = torch.zeros_like(boxes)
            new_boxes[:, 0] = torch.min(rotated_corners[:, :, 0], dim=1)[0]  # x1
            new_boxes[:, 1] = torch.min(rotated_corners[:, :, 1], dim=1)[0]  # y1
            new_boxes[:, 2] = torch.max(rotated_corners[:, :, 0], dim=1)[0]  # x2
            new_boxes[:, 3] = torch.max(rotated_corners[:, :, 1], dim=1)[0]  # y2
            
            # Clip boxes to image boundaries
            new_boxes[:, 0].clamp_(min=0, max=width)
            new_boxes[:, 1].clamp_(min=0, max=height)
            new_boxes[:, 2].clamp_(min=0, max=width)
            new_boxes[:, 3].clamp_(min=0, max=height)
            
            # Filter out invalid boxes
            valid_boxes = (new_boxes[:, 2] > new_boxes[:, 0]) & (new_boxes[:, 3] > new_boxes[:, 1])
            
            if valid_boxes.sum() > 0:
                target["boxes"] = new_boxes[valid_boxes]
                target["labels"] = target["labels"][valid_boxes]
                target["area"] = (new_boxes[valid_boxes][:, 2] - new_boxes[valid_boxes][:, 0]) * \
                                (new_boxes[valid_boxes][:, 3] - new_boxes[valid_boxes][:, 1])
                if "iscrowd" in target:
                    target["iscrowd"] = target["iscrowd"][valid_boxes]
            else:
               # sometimes fails so js pass it
                pass
                
        return image, target


class ColorJitter:
 
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(self, image, target):
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness) if self.brightness > 0 else 1
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast) if self.contrast > 0 else 1
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation) if self.saturation > 0 else 1
        hue_factor = random.uniform(-self.hue, self.hue) if self.hue > 0 else 0
        
        image = F.adjust_brightness(image, brightness_factor)
        image = F.adjust_contrast(image, contrast_factor)
        image = F.adjust_saturation(image, saturation_factor)
        image = F.adjust_hue(image, hue_factor)
        
        return image, target


class Resize:
 
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        self.max_size = max_size
        
    def __call__(self, image, target):
        # Original size
        width, height = image.size
        
        # Calculate new size
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        
        if self.max_size is not None:
            # Calculate scale factor
            if max_original_size / min_original_size * self.min_size > self.max_size:
                size = int(round(max_original_size * self.max_size / max_original_size))
            else:
                size = int(round(min_original_size * self.min_size / min_original_size))
        else:
            size = self.min_size
            
        # Calculate scale factors
        scale_width = size / width
        scale_height = size / height
        
        # Resize image
        image = F.resize(image, (int(height * scale_height), int(width * scale_width)))
        
        # Resize bounding boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            boxes[:, 0] = boxes[:, 0] * scale_width
            boxes[:, 1] = boxes[:, 1] * scale_height
            boxes[:, 2] = boxes[:, 2] * scale_width
            boxes[:, 3] = boxes[:, 3] * scale_height
            target["boxes"] = boxes
            
            # Update area
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
        return image, target


def get_transforms(config):
   
    # Training transforms with data augmentation
    train_transforms = Compose([
        RandomHorizontalFlip(config.AUGMENTATIONS['horizontal_flip']),
        RandomVerticalFlip(config.AUGMENTATIONS['vertical_flip']),
        RandomRotation(config.AUGMENTATIONS['rotation']),
        ColorJitter(
            brightness=config.AUGMENTATIONS['brightness'],
            contrast=config.AUGMENTATIONS['contrast'],
            saturation=config.AUGMENTATIONS['saturation'],
            hue=config.AUGMENTATIONS['hue']
        ),
        ToTensor(),
        Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
    ])
    
    # Validation transforms without data augmentation
    val_transforms = Compose([
        ToTensor(),
        Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
    ])
    
    return {
        'train': train_transforms,
        'val': val_transforms
    }
