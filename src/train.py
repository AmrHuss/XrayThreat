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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from config.model_config import *
from src.data.dataset import create_data_loaders
from src.models.faster_rcnn import create_faster_rcnn_model
from src.models.utils import save_model, load_model, get_latest_checkpoint, get_best_checkpoint, print_model_summary
from src.utils.metrics import MetricsTracker, compute_metrics_for_all_classes, compute_confusion_matrix
from src.utils.visualization import visualize_batch, visualize_predictions
from src.utils.logger import Logger, AverageMeter, ProgressMeter, get_timestamp


def parse_args():
 
    parser = argparse.ArgumentParser(description='Train X-ray threat detection model')
    parser.add_argument('--config', type=str, default='config/model_config.py',
                        help='Path to configuration file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable TensorBoard logging')
    parser.add_argument('--eval-only', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--ablation', type=str, default=None,
                        help='Run ablation study (backbone, anchors, optimizer)')
    return parser.parse_args()


def set_seed(seed):
    """
    Set random seed for reproducibility
    
  
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, optimizer, data_loader, device, epoch, logger, metrics_tracker, config):
    """
    Train the model for one epoch.
    
   
 
    """
    # Set model to training mode
    model.train()
    
    # Initialize meters
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses = AverageMeter('Loss', ':.4f')
    
    # Initialize progress meter
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses],
        prefix=f"Epoch: [{epoch}]"
    )
    
    current_lr = optimizer.param_groups[0]['lr']
    
    end = time.time()
    
    # Iterate over batches
    for batch_idx, (images, targets, metadata) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move data to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        loss_dict = model(images, targets)
        
        losses_sum = sum(loss for loss in loss_dict.values())
        loss_value = losses_sum.item()
        
        losses_sum.backward()
        
        # Update weights
        optimizer.step()
        
        # Update meters
        losses.update(loss_value)
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log batch
        if batch_idx % 10 == 0:
            logger.log_batch(
                epoch, batch_idx, loss_value, current_lr,
                batch_time.val, data_time.val, len(data_loader)
            )
            
            metrics_tracker.update_train_loss(loss_value, epoch, batch_idx)
        
        # Save checkpoint
        if (batch_idx + 1) % config.CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINTS_DIR,
                f'model_epoch_{epoch}_batch_{batch_idx}.pth'
            )
            save_model(model, optimizer, epoch, batch_idx, loss_value, config)
            logger.log_checkpoint(epoch, batch_idx, checkpoint_path)
        
        # Generate detection examples
        if (batch_idx + 1) % config.VISUALIZATION_INTERVAL == 0:

            model.eval()
            
            # Generate predictions
            with torch.no_grad():
                predictions = model(images)
            
            # Visualize batch
            vis_dir = os.path.join(
                config.VISUALIZATIONS_DIR,
                f'epoch_{epoch}_batch_{batch_idx}'
            )
            os.makedirs(vis_dir, exist_ok=True)
            
            visualize_batch(
                images, targets, predictions,
                class_names=config.CLASSES,
                threshold=config.VISUALIZATION_CONF_THRESHOLD,
                max_images=min(4, len(images)),
                save_dir=vis_dir
            )
            
            logger.log_visualization(epoch, batch_idx, vis_dir)
            
            # Setss model back to training mode
            model.train()
    
    # Update epoch metrics
    metrics_tracker.update_epoch_train_loss(losses.avg, epoch)
    
    return losses.avg


def evaluate(model, data_loader, device, epoch, logger, metrics_tracker, config):
    """
    Evaluates  the model.
    

        
  
    """
    # Set model to evaluation mode
    model.eval()
    
    # Init the  meters
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses = AverageMeter('Loss', ':.4f')
    
    # Initialize sprogress meter
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses],
        prefix=f"Validation: "
    )
    

    end = time.time()
    
    all_predictions = []
    all_targets = []
    
    # Iterate over the batches (DO NOT TOUCH THIS) 
    with torch.no_grad():
        for batch_idx, (images, targets, metadata) in enumerate(data_loader):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass with targets for loss computation
            loss_dict = model(images, targets)
            
            if isinstance(loss_dict, dict):
                losses_sum = sum(loss for loss in loss_dict.values())
                loss_value = losses_sum.item()
                losses.update(loss_value)
            
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
    
    #  confusion matrix 
    cm = compute_confusion_matrix(
        all_predictions, all_targets, len(config.CLASSES)
    )
    
    # Update 
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
    
    # Log 
    logger.log_metrics(epoch, metrics)
    logger.log_confusion_matrix(epoch, cm, config.CLASSES)
    
    # Save 
    metrics_tracker.save_metrics(
        os.path.join(config.METRICS_DIR, f'metrics_epoch_{epoch}.json')
    )
    
    # Visualize validation examples
    vis_dir = os.path.join(config.VISUALIZATIONS_DIR, f'val_epoch_{epoch}')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 
    images, targets, _ = next(iter(data_loader))
    
    # Gnerates the  predictions
    with torch.no_grad():
        predictions = model([img.to(device) for img in images])
    
    # Visualizes the batch
    visualize_batch(
        images, targets, predictions,
        class_names=config.CLASSES,
        threshold=config.VISUALIZATION_CONF_THRESHOLD,
        max_images=min(4, len(images)),
        save_dir=vis_dir
    )
    
    logger.log_visualization(epoch, 0, vis_dir)
    
    return losses.avg, metrics['mAP']


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Create a logger 
    logger = Logger(config=sys.modules[__name__], tensorboard=not args.no_tensorboard)
    logger.log_config()
    
    # Create data loaders
    data_loaders = create_data_loaders(config=sys.modules[__name__])
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    class_map = data_loaders['class_map']
    
    # Create model
    model = create_faster_rcnn_model(config=sys.modules[__name__])
    model.to(device)
    
    # Print model summary
    print_model_summary(model)
    logger.log_model_summary(model)
    
    # Creating all the stuff needed
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
 
    scheduler = StepLR(
        optimizer,
        step_size=LR_SCHEDULER_STEP_SIZE,
        gamma=LR_SCHEDULER_GAMMA
    )
    
   
    metrics_tracker = MetricsTracker(config=sys.modules[__name__], class_names=CLASSES)
    
    # Resume from checkpoint if requested (cheka arg path if on failuire)
    start_epoch = 0
    best_map = 0.0
    
    if args.resume:
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = get_latest_checkpoint(CHECKPOINTS_DIR)
        
        if checkpoint_path:
            model, optimizer, start_epoch, batch_idx, loss = load_model(
                model, optimizer, checkpoint_path
            )
            logger.logger.info(f"Resumed from checkpoint: {checkpoint_path}")
            logger.logger.info(f"Starting from epoch {start_epoch + 1}")
        else:
            logger.logger.warning("No checkpoint found. Starting from scratch.")
    
    # Evaluation 
    if args.eval_only:
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = get_best_checkpoint(CHECKPOINTS_DIR)
        
        if checkpoint_path:
            model, _, _, _, _ = load_model(model, None, checkpoint_path)
            logger.logger.info(f"Loaded checkpoint for evaluation: {checkpoint_path}")
        else:
            logger.logger.warning("No checkpoint found for evaluation. Using initial model.")
        
        val_loss, val_map = evaluate(
            model, val_loader, device, 0, logger, metrics_tracker, sys.modules[__name__]
        )
        logger.logger.info(f"Validation Loss: {val_loss:.4f}, Validation mAP: {val_map:.4f}")
        return
    
    # Ablation 
    if args.ablation:
        from src.models.faster_rcnn import create_model_for_ablation
        
        if args.ablation == 'backbone':
           
            backbones = ABLATION_STUDIES['backbones']
            results = {}
            
            for backbone in backbones:
                logger.logger.info(f"Running ablation study for backbone: {backbone}")
                
             
                model = create_model_for_ablation(
                    config=sys.modules[__name__],
                    backbone_name=backbone
                )
                model.to(device)
                
                
                val_loss, val_map = evaluate(
                    model, val_loader, device, 0, logger, metrics_tracker, sys.modules[__name__]
                )
                
                # Stores results
                results[backbone] = {
                    'val_loss': val_loss,
                    'val_map': val_map
                }
                
                logger.logger.info(f"Backbone: {backbone}, Validation Loss: {val_loss:.4f}, Validation mAP: {val_map:.4f}")
            
            
            logger.logger.info("Ablation Study Results (Backbone):")
            for backbone, result in results.items():
                logger.logger.info(f"  {backbone}: Loss={result['val_loss']:.4f}, mAP={result['val_map']:.4f}")
            
            return
        
        elif args.ablation == 'anchors':
            # Ablation study for different anchor configurations
            anchor_sizes = ABLATION_STUDIES['anchor_sizes']
            anchor_ratios = ABLATION_STUDIES['anchor_ratios']
            results = {}
            
            for sizes in anchor_sizes:
                for ratios in anchor_ratios:
                    logger.logger.info(f"Running ablation study for anchors: sizes={sizes}, ratios={ratios}")
                    
                    # Create model with different anchor configuration
                    model = create_model_for_ablation(
                        config=sys.modules[__name__],
                        anchor_sizes=sizes,
                        anchor_ratios=ratios
                    )
                    model.to(device)
                    
                    # Evaluate model
                    val_loss, val_map = evaluate(
                        model, val_loader, device, 0, logger, metrics_tracker, sys.modules[__name__]
                    )
                    
                    # Store results
                    key = f"sizes={sizes}, ratios={ratios}"
                    results[key] = {
                        'val_loss': val_loss,
                        'val_map': val_map
                    }
                    
                    logger.logger.info(f"Anchors: {key}, Validation Loss: {val_loss:.4f}, Validation mAP: {val_map:.4f}")
            
            # Log results
            logger.logger.info("Ablation Study Results (Anchors):")
            for key, result in results.items():
                logger.logger.info(f"  {key}: Loss={result['val_loss']:.4f}, mAP={result['val_map']:.4f}")
            
            return
        
        elif args.ablation == 'optimizer':
            #  arrays fro the optimizers 
            optimizers = ABLATION_STUDIES['optimizers']
            learning_rates = ABLATION_STUDIES['learning_rates']
            results = {}
            
            for opt_name in optimizers:
                for lr in learning_rates:
                    logger.logger.info(f"Running ablation study for optimizer: {opt_name}, lr={lr}")
                    
                    # Create optimizer
                    if opt_name == 'SGD':
                        optimizer = optim.SGD(
                            model.parameters(),
                            lr=lr,
                            momentum=MOMENTUM,
                            weight_decay=WEIGHT_DECAY
                        )
                    elif opt_name == 'Adam':
                        optimizer = optim.Adam(
                            model.parameters(),
                            lr=lr,
                            weight_decay=WEIGHT_DECAY
                        )
                    elif opt_name == 'AdamW':
                        optimizer = optim.AdamW(
                            model.parameters(),
                            lr=lr,
                            weight_decay=WEIGHT_DECAY
                        )
                    else:
                        logger.logger.warning(f"Unknown optimizer: {opt_name}. Skipping.")
                        continue
                    
                    
                    for epoch in range(3):
                        train_loss = train_one_epoch(
                            model, optimizer, train_loader, device, epoch,
                            logger, metrics_tracker, sys.modules[__name__]
                        )
                        
                        val_loss, val_map = evaluate(
                            model, val_loader, device, epoch, logger, metrics_tracker, sys.modules[__name__]
                        )
                        
                        logger.logger.info(
                            f"Optimizer: {opt_name}, LR: {lr}, "
                            f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, "
                            f"Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}"
                        )
                    
                    # nesting //fix before submit 
                    key = f"{opt_name}, lr={lr}"
                    results[key] = {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_map': val_map
                    }
            
            # Log results
            logger.logger.info("Ablation Study Results (Optimizer):")
            for key, result in results.items():
                logger.logger.info(
                    f"  {key}: Train Loss={result['train_loss']:.4f}, "
                    f"Val Loss={result['val_loss']:.4f}, Val mAP={result['val_map']:.4f}"
                )
            
            return
        
        else:
            logger.logger.warning(f"Unknown ablation study: {args.ablation}")
            return
    
    # Training loop
    logger.logger.info("Starting training")
    start_time = time.time()
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Train for one epoch
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            logger, metrics_tracker, sys.modules[__name__]
        )
        
        # Evaluate
        val_loss, val_map = evaluate(
            model, val_loader, device, epoch, logger, metrics_tracker, sys.modules[__name__]
        )
        
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        logger.log_lr_scheduler(epoch, current_lr)
        metrics_tracker.update_learning_rate(current_lr, epoch)
        
        save_model(model, optimizer, epoch, 0, val_loss, sys.modules[__name__])
        
        # Save best models IN THE EPOCH SAVE MODEL BEFORE or it deletes  
        if val_map > best_map:
            best_map = val_map
            save_model(model, optimizer, epoch, 0, val_loss, sys.modules[__name__], is_best=True)
            logger.log_best_model(epoch, val_map, os.path.join(CHECKPOINTS_DIR, 'model_best.pth'))
        
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
