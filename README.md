# X-ray Automated Threat Detection System

This repository contains the implementation of an X-ray automated threat detection system using Faster R-CNN with ResNet50 backbone. The system is designed to detect various threats in X-ray images, including different types of knives and scissors.


### Key Features

- Faster R-CNN with ResNet50 backbone for object detection
- Multi-class threat detection (Straight_Knife, Folding_Knife, Utility_Knife, Multi-tool_Knife, Scissor)
- Periodic model checkpoint saving (every 100 batches)
- Test detection visualization (every 50 batches)
- Comprehensive metrics tracking and visualization
- Ablation studies for model components
- Confusion matrix analysis for threat classes

## Directory Structure


#Installation

1. Clone the repository:
```bash
git clone https://github.com/AmrHuss/Xray-automated-threat-detection.git # Will Upload repo soon.
cd XrayDetector
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate 
```

3. Install the required dependencies: ## sd
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

The system expects data in the following format:
- Images in the `train_image/` directory
- Annotations in the `train_annotation/` directory with format: `[image_filename] [class_name] [x1] [y1] [x2] [y2]`

### Training

## PLease read the args of the file you are running it explains the args 
To train the model:

```bash
python src/train.py --config config/model_config.py 
or
python src/train.py --checkpoint checkpoint/....
```

Training parameters can be modified in the configuration file.

### Evaluation

To evaluate the trained model:

```bash
python src/evaluate.py --model_path checkpoints/model_final.pth
```

### Prediction

To run predictions on new images:

```bash
python src/predict.py --image_path path/to/image.jpg --model_path checkpoints/model_final.pth
```

## Model Architecture

The system uses Faster R-CNN with ResNet50 backbone, which consists of:
1. **Backbone Network**: ResNet50 pre-trained on ImageNet
2. **Region Proposal Network (RPN)**: Generates region proposals
3. **RoI Pooling**: Extracts features from proposed regions
4. **Classification Head**: Classifies objects and refines bounding boxes

## Performance Metrics

The system tracks the following metrics:
- Mean Average Precision (mAP)
- Precision and Recall per class
- F1-score
- Confusion Matrix
- Training and Validation Loss

## Ablation Studies

The repository includes code for ablation studies to analyze:
- Impact of different backbones
- Effect of anchor sizes and aspect ratios
- Influence of data augmentation techniques
- Comparison of optimization strategies


