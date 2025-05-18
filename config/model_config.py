"""
Configuration file for the X-ray Threat Detection model.
"""

import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, 'train_image')
TRAIN_ANNOTATIONS_DIR = os.path.join(BASE_DIR, 'train_annotation')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
VISUALIZATIONS_DIR = os.path.join(RESULTS_DIR, 'visualizations')

# Ensure directories exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Run identifier
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

# Dataset configuration
CLASSES = ['background', 'Straight_Knife', 'Folding_Knife', 'Utility_Knife', 'Multi-tool_Knife', 'Scissor']
NUM_CLASSES = len(CLASSES)
TRAIN_VAL_SPLIT = 0.95  # 80% training, 20% validation
RANDOM_SEED = 42

# Model configuration
BACKBONE = 'resnet50'
PRETRAINED = True
TRAINABLE_BACKBONE_LAYERS = 3

# FasterRCNN configuration
MIN_SIZE = 800
MAX_SIZE = 1333
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
RPN_ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))
RPN_ANCHOR_RATIOS = ((0.5, 1.0, 2.0),)
RPN_PRE_NMS_TOP_N_TRAIN = 2000
RPN_PRE_NMS_TOP_N_TEST = 1000
RPN_POST_NMS_TOP_N_TRAIN = 2000
RPN_POST_NMS_TOP_N_TEST = 1000
RPN_NMS_THRESH = 0.7
RPN_FG_IOU_THRESH = 0.7
RPN_BG_IOU_THRESH = 0.3
RPN_BATCH_SIZE_PER_IMAGE = 256
RPN_POSITIVE_FRACTION = 0.5
BOX_SCORE_THRESH = 0.05
BOX_NMS_THRESH = 0.5
BOX_DETECTIONS_PER_IMG = 100
BOX_FG_IOU_THRESH = 0.5
BOX_BG_IOU_THRESH = 0.5
BOX_BATCH_SIZE_PER_IMAGE = 512
BOX_POSITIVE_FRACTION = 0.25

# Training configuration
BATCH_SIZE = 4
NUM_WORKERS = 2
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 20
WARMUP_EPOCHS = 1
LR_SCHEDULER_GAMMA = 0.1
LR_SCHEDULER_STEP_SIZE = 5

# Checkpoint and evaluation configuration
CHECKPOINT_INTERVAL = 800  
VISUALIZATION_INTERVAL = 50  
EVAL_INTERVAL = 1  

# Data augmentation
AUGMENTATIONS = {
    'horizontal_flip': 0.5,  
    'vertical_flip': 0.0,    
    'rotation': 15,          
    'brightness': 0.2,       
    'contrast': 0.2,         
    'saturation': 0.2,      
    'hue': 0.1,              
}

# Ablation study configurations
ABLATION_STUDIES = {
    'backbones': ['resnet18', 'resnet34', 'resnet50', 'resnet101'],
    'anchor_sizes': [
        ((16,), (32,), (64,), (128,), (256,)),
        ((32,), (64,), (128,), (256,), (512,)),
        ((64,), (128,), (256,), (512,), (1024,)),
    ],
    'anchor_ratios': [
        ((0.5, 1.0, 2.0),),
        ((0.3, 1.0, 3.0),),
        ((0.5, 1.0, 2.0, 3.0),),
    ],
    'optimizers': ['SGD', 'Adam', 'AdamW'],
    'learning_rates': [0.001, 0.005, 0.01],
}

# Visualization settings
VISUALIZATION_CONF_THRESHOLD = 0.6  # Cahnged from 0.3->0.6 Confidence threshold for visualization
VISUALIZATION_MAX_DETECTIONS = 10   