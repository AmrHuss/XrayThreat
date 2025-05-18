
import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from datetime import datetime

# Add src directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import  modules
from config.model_config import *
from src.models.faster_rcnn import create_faster_rcnn_model
from src.models.utils import load_model, get_best_checkpoint, get_inference_time
from src.utils.visualization import visualize_detection
from src.utils.logger import get_timestamp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict with X-ray threat detection model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to use')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output image')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--measure-time', action='store_true',
                        help='Measure inference time')
    return parser.parse_args()


def preprocess_image(image_path):
  
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Convert to tensor(double check with the documentation)
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    # Normalize
    image_tensor = image_tensor.sub(torch.tensor(IMAGE_MEAN).view(3, 1, 1)).div(torch.tensor(IMAGE_STD).view(3, 1, 1))
    
    return image_tensor


def predict(model, image_tensor, device, threshold=0.5):
    """
    Make prediction on an image
    

    """
    # Set model to evaluation mode
    model.eval()
    
    # Move image to device
    image_tensor = image_tensor.to(device)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    
    # Filter predictions by threshold
    keep = prediction['scores'] > threshold
    
    filtered_prediction = {
        'boxes': prediction['boxes'][keep],
        'labels': prediction['labels'][keep],
        'scores': prediction['scores'][keep]
    }
    
    return filtered_prediction


def main():
   
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Create model
    model = create_faster_rcnn_model(config=sys.modules[__name__])
    model.to(device)
    
    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = get_best_checkpoint(CHECKPOINTS_DIR)
    
    if checkpoint_path:
        model, _, _, _, _ = load_model(model, None, checkpoint_path)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found. Using initial model.")
    
    # Preprocess image
    image_tensor = preprocess_image(args.image)
    
    # Measure inference time
    if args.measure_time:
        inference_time = get_inference_time(
            model, image_tensor.unsqueeze(0), num_runs=100, warmup_runs=10
        )
        print(f"Inference time: {inference_time:.2f} ms")
    
    # Make prediction
    start_time = time.time()
    prediction = predict(model, image_tensor, device, args.threshold)
    end_time = time.time()
    
    print(f"Prediction time: {(end_time - start_time) * 1000:.2f} ms")
    
    # Print prediction results
    print(f"Found {len(prediction['boxes'])} threats:")
    for i, (box, label, score) in enumerate(zip(prediction['boxes'], prediction['labels'], prediction['scores'])):
        class_name = CLASSES[label.item()]
        print(f"  {i+1}. {class_name}: {score.item():.4f} at {box.tolist()}")
    
    # Visualize prediction
    image = Image.open(args.image).convert('RGB')
    
    vis_image = visualize_detection(
        image,
        prediction['boxes'].cpu(),
        prediction['labels'].cpu(),
        prediction['scores'].cpu(),
        class_names=CLASSES,
        threshold=args.threshold
    )
    
    # Save or show visualization
    if args.output:
        output_path = args.output
    else:
        timestamp = get_timestamp()
        output_dir = os.path.join(RESULTS_DIR, 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'prediction_{timestamp}.png')
    
    # Convert visualization to BGR for OpenCV
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGBA2BGR)
    
    # Save visualization
    cv2.imwrite(output_path, vis_image)
    print(f"Visualization saved to {output_path}")


if __name__ == '__main__':
    main()
