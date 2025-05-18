

import os
import sys
import logging
import time
from datetime import datetime
import json
import torch
from torch.utils.tensorboard import SummaryWriter

# 
class Logger:
   
    def __init__(self, config, log_dir=None, tensorboard=True):
    
        self.config = config
        
        # Set log directory
        if log_dir is None:
            self.log_dir = os.path.join(config.RESULTS_DIR, 'logs')
        else:
            self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging
        self.setup_logging()
        
        # Set up TensorBoard
        self.tensorboard = tensorboard
        if tensorboard:
            self.tb_dir = os.path.join(self.log_dir, 'tensorboard')
            os.makedirs(self.tb_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tb_dir)
    
    def setup_logging(self):
        """Set up logging."""
        # Create logger
        self.logger = logging.getLogger('xray_threat_detection')
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Create file handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'log_{timestamp}.txt')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Save log file path
        self.log_file = log_file
        
        # Log start message
        self.logger.info(f"Logging to {log_file}")
        self.logger.info(f"TensorBoard logs will be saved to {os.path.join(self.log_dir, 'tensorboard')}")
    
    def log_config(self):
        """Log configuration."""
        self.logger.info("Configuration:")
        
        # Get configuration as dictionary
        config_dict = {}
        for key in dir(self.config):
            if not key.startswith('__') and not callable(getattr(self.config, key)):
                value = getattr(self.config, key)
                
                # Convert non-serializable values to strings
                if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], (list, tuple)):
                    value = str(value)
                
                config_dict[key] = value
        
        # Log configuration as JSON
        config_json = json.dumps(config_dict, indent=4, default=str)
        self.logger.info(f"\n{config_json}")
        
        # Save configuration to file
        config_file = os.path.join(self.log_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
        
        self.logger.info(f"Configuration saved to {config_file}")
    
    def log_model_summary(self, model):
       
        self.logger.info("Model Summary:")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Log parameter counts
        self.logger.info(f"Total Parameters: {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")
        self.logger.info(f"Non-Trainable Parameters: {total_params - trainable_params:,}")
        
        # Log model architecture
        self.logger.info("Model Architecture:")
        self.logger.info(f"\n{model}")
    
    def log_batch(self, epoch, batch_idx, loss, lr, batch_time, data_time, total_batches):
       
        # Compute progress
        progress = 100.0 * batch_idx / total_batches
        
        # Log to console and file
        self.logger.info(
            f"Epoch: [{epoch}][{batch_idx}/{total_batches}] "
            f"Progress: {progress:.1f}% "
            f"Loss: {loss:.4f} "
            f"LR: {lr:.6f} "
            f"Time: {batch_time:.3f}s "
            f"Data: {data_time:.3f}s"
        )
        
        # Log to TensorBoard
        if self.tensorboard:
            # Compute global step
            global_step = epoch * total_batches + batch_idx
            
            # Log scalars
            self.writer.add_scalar('train/loss', loss, global_step)
            self.writer.add_scalar('train/learning_rate', lr, global_step)
            self.writer.add_scalar('train/batch_time', batch_time, global_step)
            self.writer.add_scalar('train/data_time', data_time, global_step)
    
    def log_epoch(self, epoch, train_loss, val_loss, val_map, class_aps, epoch_time):
      
        # Log to console and file
        self.logger.info(
            f"Epoch: {epoch} "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val mAP: {val_map:.4f} "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Log class APs
        self.logger.info("Class APs:")
        for class_name, ap in class_aps.items():
            self.logger.info(f"  {class_name}: {ap:.4f}")
        
        # Log to TensorBoard
        if self.tensorboard:
            # Log scalars
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/val_map', val_map, epoch)
            self.writer.add_scalar('epoch/epoch_time', epoch_time, epoch)
            
            # Log class APs
            for class_name, ap in class_aps.items():
                self.writer.add_scalar(f'epoch/class_ap/{class_name}', ap, epoch)
    
    def log_checkpoint(self, epoch, batch_idx, checkpoint_path):
     
        self.logger.info(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}: {checkpoint_path}")
    
    def log_best_model(self, epoch, val_map, checkpoint_path):
    
        self.logger.info(f"New best model at epoch {epoch} with mAP {val_map:.4f}: {checkpoint_path}")
    
    def log_visualization(self, epoch, batch_idx, vis_path):
        
        self.logger.info(f"Visualization saved at epoch {epoch}, batch {batch_idx}: {vis_path}")
        
        # Log to TensorBoard
        if self.tensorboard:
            # Compute global step
            global_step = epoch * self.config.BATCH_SIZE + batch_idx
            
            # Add image to TensorBoard
            try:
                import cv2
                import numpy as np
                
                # Read image
                image = cv2.imread(vis_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Add image to TensorBoard
                self.writer.add_image(
                    f'visualization/epoch_{epoch}_batch_{batch_idx}',
                    np.transpose(image, (2, 0, 1)),
                    global_step
                )
            except Exception as e:
                self.logger.warning(f"Failed to add visualization to TensorBoard: {e}")
    
    def log_metrics(self, epoch, metrics):
     
        # Log to console and file
        self.logger.info(f"Metrics for epoch {epoch}:")
        
        # Log mAP
        if 'mAP' in metrics:
            self.logger.info(f"  mAP: {metrics['mAP']:.4f}")
        
        # Log class APs
        if 'class_APs' in metrics:
            self.logger.info("  Class APs:")
            for class_name, ap in metrics['class_APs'].items():
                self.logger.info(f"    {class_name}: {ap:.4f}")
        
        # Log precision, recall, and F1 score
        if 'precision' in metrics and 'recall' in metrics and 'f1_score' in metrics:
            self.logger.info("  Precision, Recall, and F1 Score:")
            for class_name in metrics['precision'].keys():
                self.logger.info(f"    {class_name}:")
                self.logger.info(f"      Precision: {metrics['precision'][class_name]:.4f}")
                self.logger.info(f"      Recall: {metrics['recall'][class_name]:.4f}")
                self.logger.info(f"      F1 Score: {metrics['f1_score'][class_name]:.4f}")
        
        # Log inference time
        if 'inference_time' in metrics:
            self.logger.info(f"  Inference Time: {metrics['inference_time']:.2f} ms")
        
        # Log to TensorBoard
        if self.tensorboard:
            # Log mAP
            if 'mAP' in metrics:
                self.writer.add_scalar('metrics/mAP', metrics['mAP'], epoch)
            
            # Log class APs
            if 'class_APs' in metrics:
                for class_name, ap in metrics['class_APs'].items():
                    self.writer.add_scalar(f'metrics/class_ap/{class_name}', ap, epoch)
            
            # Log precision, recall, and F1 score
            if 'precision' in metrics and 'recall' in metrics and 'f1_score' in metrics:
                for class_name in metrics['precision'].keys():
                    self.writer.add_scalar(f'metrics/precision/{class_name}', metrics['precision'][class_name], epoch)
                    self.writer.add_scalar(f'metrics/recall/{class_name}', metrics['recall'][class_name], epoch)
                    self.writer.add_scalar(f'metrics/f1_score/{class_name}', metrics['f1_score'][class_name], epoch)
            
            # Log inference time
            if 'inference_time' in metrics:
                self.writer.add_scalar('metrics/inference_time', metrics['inference_time'], epoch)
    
    def log_confusion_matrix(self, epoch, cm, class_names):
    
        # Log to console and file
        self.logger.info(f"Confusion Matrix for epoch {epoch}:")
        
        # Format confusion matrix as string
        cm_str = "\n"
        cm_str += "  " + " ".join(f"{name:>10}" for name in class_names[1:]) + "\n"
        
        for i, row in enumerate(cm):
            cm_str += f"{class_names[i+1]:>2} " + " ".join(f"{val:>10}" for val in row) + "\n"
        
        self.logger.info(cm_str)
        
        # Log to TensorBoard
        if self.tensorboard:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                import numpy as np
                
                # Create figure
                plt.figure(figsize=(10, 8))
                
                # Plot confusion matrix
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=class_names[1:],
                    yticklabels=class_names[1:]
                )
                
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix (Epoch {epoch})')
                
                # Convert figure to numpy array
                plt.tight_layout()
                fig = plt.gcf()
                fig.canvas.draw()
                cm_image = np.array(fig.canvas.renderer.buffer_rgba())
                plt.close()
                
                # Add image to TensorBoard
                self.writer.add_image(
                    f'confusion_matrix/epoch_{epoch}',
                    np.transpose(cm_image[:, :, :3], (2, 0, 1)),
                    epoch
                )
            except Exception as e:
                self.logger.warning(f"Failed to add confusion matrix to TensorBoard: {e}")
    
    def log_lr_scheduler(self, epoch, lr):
      
        self.logger.info(f"Learning rate at epoch {epoch}: {lr:.6f}")
        
        # Log to TensorBoard
        if self.tensorboard:
            self.writer.add_scalar('train/learning_rate', lr, epoch)
    
    def log_early_stopping(self, epoch, patience, best_epoch, best_map):
       
        self.logger.info(
            f"Early stopping: No improvement for {epoch - best_epoch} epochs. "
            f"Best mAP: {best_map:.4f} at epoch {best_epoch}. "
            f"Patience: {patience}."
        )
    
    def log_training_complete(self, epochs, best_epoch, best_map, total_time):
      
        # Format time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        
        # Log to console and file
        self.logger.info(f"Training complete after {epochs} epochs.")
        self.logger.info(f"Best mAP: {best_map:.4f} at epoch {best_epoch}.")
        self.logger.info(f"Total training time: {time_str}")
        
        # Log to TensorBoard
        if self.tensorboard:
            self.writer.add_text('training_summary', f"Training complete after {epochs} epochs.")
            self.writer.add_text('training_summary', f"Best mAP: {best_map:.4f} at epoch {best_epoch}.")
            self.writer.add_text('training_summary', f"Total training time: {time_str}")
    
    def close(self):
        """Close the logger."""
         
        if self.tensorboard:
            self.writer.close()
        
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class AverageMeter:
   
    
    def __init__(self, name, fmt=':f'):
        
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        """Reset the meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
       
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        """Return string representation."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    
    
    def __init__(self, num_batches, meters, prefix=""):
        
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch):
        
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)
    
    def _get_batch_fmtstr(self, num_batches):
      
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_timestamp():

    return datetime.now().strftime('%Y%m%d_%H%M%S')# python doing pythong things //delete
