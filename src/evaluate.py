

import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import  modules
from config.model_config import *
from src.data.dataset import create_data_loaders
from src.models.faster_rcnn import create_faster_rcnn_model
from src.models.utils import load_model, get_best_checkpoint, get_model_size, get_inference_time
from src.utils.metrics import MetricsTracker, compute_metrics_for_all_classes, compute_confusion_matrix
from src.utils.visualization import visualize_batch, visualize_predictions, visualize_feature_maps, visualize_class_activation_map
from src.utils.logger import Logger, AverageMeter, ProgressMeter, get_timestamp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate X-ray threat detection model')
    parser.add_argument('--config', type=str, default='config/model_config.py',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--feature-maps', action='store_true',
                        help='Visualize feature maps')
    parser.add_argument('--cam', action='store_true',
                        help='Visualize class activation maps')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save evaluation results')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to visualize')
    return parser.parse_args()


def set_seed(seed):
  
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_model(model, data_loader, device, logger, metrics_tracker, config, args):
    """
    Evaluate the model
    
    
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize meters
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    

    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time],
        prefix='Evaluation: '
    )
    

    end = time.time()
    

    all_predictions = []
    all_targets = []
    

    with torch.no_grad():
        for batch_idx, (images, targets, metadata) in enumerate(data_loader):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            predictions = model(images)
            
            # Store predictions and targets for metrics computation
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
         
            if batch_idx % 10 == 0:
                logger.logger.info(progress.display(batch_idx))
            
        
            if args.visualize and batch_idx < args.num_samples:
            
                vis_dir = os.path.join(args.output_dir, 'visualizations', f'batch_{batch_idx}')
                os.makedirs(vis_dir, exist_ok=True)
                
                # Visualize 
                visualize_batch(
                    images, targets, predictions,
                    class_names=config.CLASSES,
                    threshold=config.VISUALIZATION_CONF_THRESHOLD,
                    max_images=min(4, len(images)),
                    save_dir=vis_dir
                )
                
                # Visualize feature maps
                if args.feature_maps:
                    # Set model to evaluation mode
                    model.eval()
                    
                    # Create feature maps directory
                    feature_maps_dir = os.path.join(args.output_dir, 'feature_maps', f'batch_{batch_idx}')
                    os.makedirs(feature_maps_dir, exist_ok=True)
                    
                    # Visualize feature maps for the first image
                    for name, module in model.named_modules():
                        if 'backbone' in name and 'layer' in name and len(name.split('.')) == 3:
                            visualize_feature_maps(
                                model, images[0], name,
                                save_dir=feature_maps_dir
                            )
                
                # Visualize class activation maps
                if args.cam:
                    # Set model to evaluation mode
                    model.eval()
                    
                    
                    cam_dir = os.path.join(args.output_dir, 'cam', f'batch_{batch_idx}')
                    os.makedirs(cam_dir, exist_ok=True)
                    
                    
                    for i, (pred, target) in enumerate(zip(predictions[:1], targets[:1])):
                        for j, (box, label, score) in enumerate(zip(pred['boxes'], pred['labels'], pred['scores'])):
                            if score > config.VISUALIZATION_CONF_THRESHOLD:
                                visualize_class_activation_map(
                                    model, images[i], j, 'backbone.layer4',
                                    save_path=os.path.join(cam_dir, f'cam_{j}_{config.CLASSES[label]}.png')
                                )
    
    #  metrics
    metrics = compute_metrics_for_all_classes(
        all_predictions, all_targets, config.CLASSES
    )
    
    # Compute confusion marix
    cm = compute_confusion_matrix(
        all_predictions, all_targets, len(config.CLASSES)
    )
    
    # Update metrics
    metrics_tracker.update_mAP(metrics['mAP'], 0)
    
    for class_name, ap in metrics['class_APs'].items():
        metrics_tracker.update_class_AP(class_name, ap, 0)
    
    for class_name in metrics['precision'].keys():
        metrics_tracker.update_precision_recall_f1(
            class_name,
            metrics['precision'][class_name],
            metrics['recall'][class_name],
            metrics['f1_score'][class_name],
            0
        )
    
    metrics_tracker.update_confusion_matrix(cm, 0)
    
    # Measure inference time
    if len(images) > 0:
        inference_time = get_inference_time(
            model, images[0].unsqueeze(0), num_runs=100, warmup_runs=10
        )
        metrics_tracker.update_inference_time(inference_time, 0)
        metrics['inference_time'] = inference_time
    
    # Log metrics
    logger.log_metrics(0, metrics)
    logger.log_confusion_matrix(0, cm, config.CLASSES)
    
    # Save metrics
    metrics_tracker.save_metrics(
        os.path.join(args.output_dir, 'metrics.json')
    )
    
    # Plot metrics
    metrics_tracker.plot_all_metrics(
        save_dir=os.path.join(args.output_dir, 'plots')
    )
    
    return metrics


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu') amd gpu probkems
    
    device = torch.device('cpu')
    
    # Set output directory
    if args.output_dir is None:
        timestamp = get_timestamp()
        args.output_dir = os.path.join(RESULTS_DIR, f'evaluation_{timestamp}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create logger
    logger = Logger(config=sys.modules[__name__], log_dir=os.path.join(args.output_dir, 'logs'))
    logger.log_config()
    
    # Create data loaders
    data_loaders = create_data_loaders(config=sys.modules[__name__])
    val_loader = data_loaders['val']
    class_map = data_loaders['class_map']
    
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
        logger.logger.info(f"Loaded checkpoint for evaluation: {checkpoint_path}")
    else:
        logger.logger.warning("No checkpoint found for evaluation. Using initial model.")
    
    metrics_tracker = MetricsTracker(config=sys.modules[__name__], class_names=CLASSES)
    
    model_size = get_model_size(model)
    logger.logger.info(f"Model size: {model_size:.2f} MB")
    

    logger.logger.info("Starting evaluation")
    start_time = time.time()
    
    metrics = evaluate_model(model, val_loader, device, logger, metrics_tracker, sys.modules[__name__], args)
    
    # Log evaluation time
    evaluation_time = time.time() - start_time
    logger.logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Log metrics
    logger.logger.info(f"mAP: {metrics['mAP']:.4f}")
    logger.logger.info("Class APs:")
    for class_name, ap in metrics['class_APs'].items():
        logger.logger.info(f"  {class_name}: {ap:.4f}")
    
    if 'inference_time' in metrics:
        logger.logger.info(f"Inference time: {metrics['inference_time']:.2f} ms")
    
    #Hva to close or it will just delete itself
    logger.close()


if __name__ == '__main__':
    main()
