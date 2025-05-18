import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from datetime import datetime

# FOR COLAB TRAIN ONLY!
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)  


from config.model_config import *
from src.data.dataset import create_data_loaders
from src.models.faster_rcnn import create_faster_rcnn_model
from src.models.utils import save_model, load_model, get_latest_checkpoint, get_best_checkpoint, print_model_summary
from src.utils.metrics import MetricsTracker, compute_metrics_for_all_classes, compute_confusion_matrix
from src.utils.visualization import visualize_batch, visualize_predictions
from src.utils.logger import Logger, AverageMeter, ProgressMeter, get_timestamp

def parse_args():
    parser = argparse.ArgumentParser(description='Train X-ray threat detection model on Google Colab')
    parser.add_argument('--config', type=str, default='config/model_config.py',
                        help='Path to configuration file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default='/content/drive/MyDrive/XrayDetector/checkpoints',
                        help='Directory to save checkpoints')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, optimizer, data_loader, device, epoch, logger, metrics_tracker, config):
    """Train the model for one epoch."""
    model.train()
    
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses = AverageMeter('Loss', ':.4f')
    
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses],
        prefix=f"Epoch: [{epoch}]"
    )
    
    current_lr = optimizer.param_groups[0]['lr']
    end = time.time()
    
    for batch_idx, (images, targets, metadata) in enumerate(data_loader):
        data_time.update(time.time() - end)
        
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        loss_dict = model(images, targets)
        losses_sum = sum(loss for loss in loss_dict.values())
        loss_value = losses_sum.item()
        
        losses_sum.backward()
        optimizer.step()
        
        losses.update(loss_value)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % 10 == 0:
            logger.log_batch(
                epoch, batch_idx, loss_value, current_lr,
                batch_time.val, data_time.val, len(data_loader)
            )
            metrics_tracker.update_train_loss(loss_value, epoch, batch_idx)
        
        if (batch_idx + 1) % config.CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINTS_DIR,
                f'model_epoch_{epoch}_batch_{batch_idx}.pth'
            )
            save_model(model, optimizer, epoch, batch_idx, loss_value, config)
            logger.log_checkpoint(epoch, batch_idx, checkpoint_path)
    
    metrics_tracker.update_epoch_train_loss(losses.avg, epoch)
    return losses.avg

def evaluate(model, data_loader, device, epoch, logger, metrics_tracker, config):
    """Evaluate the model."""
    model.eval()
    
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses = AverageMeter('Loss', ':.4f')
    
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses],
        prefix=f"Validation: "
    )
    
    end = time.time()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets, metadata) in enumerate(data_loader):
            data_time.update(time.time() - end)
            
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass with targets for loss computation
            loss_dict = model(images, targets)
            
            # Compute loss if we got a dict (training mode)
            if isinstance(loss_dict, dict):
                losses_sum = sum(loss for loss in loss_dict.values())
                loss_value = losses_sum.item()
                losses.update(loss_value)
            
            # Forward pass without targets for predictions
            predictions = model(images)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx % 10 == 0:
                logger.logger.info(progress.display(batch_idx))
    
    metrics = compute_metrics_for_all_classes(
        all_predictions, all_targets, config.CLASSES
    )
    
    cm = compute_confusion_matrix(
        all_predictions, all_targets, len(config.CLASSES)
    )
    
    metrics_tracker.update_epoch_val_loss(losses.avg, epoch)
    metrics_tracker.update_mAP(metrics['mAP'], epoch)
    
    for class_name, ap in metrics['class_APs'].items():
        metrics_tracker.update_class_AP(class_name, ap, epoch)
    
    for class_name in metrics['precision'].keys():
        metrics_tracker.update_precision_recall_f1(
            class_name,
            metrics['precision'][class_name],
            metrics['recall'][class_name],
            metrics['f1_score'][class_name],
            epoch
        )
    
    metrics_tracker.update_confusion_matrix(cm, epoch)
    
    logger.log_metrics(epoch, metrics)
    logger.log_confusion_matrix(epoch, cm, config.CLASSES)
    
    metrics_tracker.save_metrics(
        os.path.join(config.METRICS_DIR, f'metrics_epoch_{epoch}.json')
    )
    
    return losses.avg, metrics['mAP']

def main():
    """Main function for Google Colab training."""
    args = parse_args()
    set_seed(args.seed)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create logger
    logger = Logger(config=sys.modules[__name__], tensorboard=True)
    logger.log_config()
    
    # Create data loaders
    data_loaders = create_data_loaders(config=sys.modules[__name__])
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    class_map = data_loaders['class_map']
    
    # Create model
    model = create_faster_rcnn_model(config=sys.modules[__name__])
    model.to(device)
    
    print_model_summary(model)
    logger.log_model_summary(model)
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    # Create learning rate scheduler
    scheduler = StepLR(
        optimizer,
        step_size=LR_SCHEDULER_STEP_SIZE,
        gamma=LR_SCHEDULER_GAMMA
    )
    
    # Create metrics tracker
    metrics_tracker = MetricsTracker(config=sys.modules[__name__], class_names=CLASSES)
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_map = 0.0
    
    if args.resume:
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = get_latest_checkpoint(args.save_dir)
        
        if checkpoint_path:
            model, optimizer, start_epoch, batch_idx, loss = load_model(
                model, optimizer, checkpoint_path
            )
            logger.logger.info(f"Resumed from checkpoint: {checkpoint_path}")
            logger.logger.info(f"Starting from epoch {start_epoch + 1}")
        else:
            logger.logger.warning("No checkpoint found. Starting from scratch.")
    
    # Training loop
    logger.logger.info("Starting training")
    start_time = time.time()
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            logger, metrics_tracker, sys.modules[__name__]
        )
        
        val_loss, val_map = evaluate(
            model, val_loader, device, epoch, logger, metrics_tracker, sys.modules[__name__]
        )
        
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        logger.log_lr_scheduler(epoch, current_lr)
        metrics_tracker.update_learning_rate(current_lr, epoch)
        
        # Save checkpoint
        save_model(model, optimizer, epoch, 0, val_loss, sys.modules[__name__])
        
        # Save best model
        if val_map > best_map:
            best_map = val_map
            save_model(model, optimizer, epoch, 0, val_loss, sys.modules[__name__], is_best=True)
            logger.log_best_model(epoch, val_map, os.path.join(args.save_dir, 'model_best.pth'))
        
        # Log epoch
        logger.log_epoch(
            epoch, train_loss, val_loss, val_map,
            metrics_tracker.get_latest_metrics()['class_APs'],
            time.time() - start_time
        )
    
    # Log training completion
    total_time = time.time() - start_time
    logger.log_training_complete(NUM_EPOCHS, epoch, best_map, total_time)
    
    # Close logger
    logger.close()

if __name__ == '__main__':
    main() 