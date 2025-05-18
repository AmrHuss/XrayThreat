
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import json
from datetime import datetime


class MetricsTracker:
   
    def __init__(self, config, class_names):
     
     
        self.config = config
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Create metrics directory if it doesn't exist
        os.makedirs(config.METRICS_DIR, exist_ok=True)
        
        # Initialize metrics
        self.reset()
    
    def reset(self):
        """Reset"""
        self.train_losses = []
        self.val_losses = []
        self.train_batch_losses = []
        self.learning_rates = []
        self.mAP_values = []
        self.class_APs = defaultdict(list)
        self.precisions = defaultdict(list)
        self.recalls = defaultdict(list)
        self.f1_scores = defaultdict(list)
        self.confusion_matrices = []
        self.inference_times = []
        
        # Epoch-wise metrics
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.epoch_mAP_values = []
        self.epoch_class_APs = defaultdict(list)
    
    def update_train_loss(self, loss, epoch, batch_idx):
        
        self.train_batch_losses.append((epoch, batch_idx, float(loss)))
    
    def update_epoch_train_loss(self, loss, epoch):
        
        self.train_losses.append((epoch, float(loss)))
        self.epoch_train_losses.append(float(loss))
    
    def update_epoch_val_loss(self, loss, epoch):
       
        self.val_losses.append((epoch, float(loss)))
        self.epoch_val_losses.append(float(loss))
    
    def update_learning_rate(self, lr, epoch):
       
        self.learning_rates.append((epoch, float(lr)))
    
    def update_mAP(self, mAP, epoch):
      
        self.mAP_values.append((epoch, float(mAP)))
        self.epoch_mAP_values.append(float(mAP))
    
    def update_class_AP(self, class_name, ap, epoch):
        
        self.class_APs[class_name].append((epoch, float(ap)))
        self.epoch_class_APs[class_name].append(float(ap))
    
    def update_precision_recall_f1(self, class_name, precision, recall, f1, epoch):
       
        self.precisions[class_name].append((epoch, float(precision)))
        self.recalls[class_name].append((epoch, float(recall)))
        self.f1_scores[class_name].append((epoch, float(f1)))
    
    def update_confusion_matrix(self, cm, epoch):
        
        self.confusion_matrices.append((epoch, cm.tolist()))
    
    def update_inference_time(self, time_ms, epoch):
       
        self.inference_times.append((epoch, float(time_ms)))
    
    def save_metrics(self, filename=None):
     
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.config.METRICS_DIR, f'metrics_{timestamp}.json')
        
       
        metrics = {
            # Need to get from cfg maybe hardcode fro nwo json.hpp 
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_batch_losses': self.train_batch_losses,
            'learning_rates': self.learning_rates,
            'mAP_values': self.mAP_values,
            'class_APs': {k: v for k, v in self.class_APs.items()},
            'precisions': {k: v for k, v in self.precisions.items()},
            'recalls': {k: v for k, v in self.recalls.items()},
            'f1_scores': {k: v for k, v in self.f1_scores.items()},
            'confusion_matrices': self.confusion_matrices,
            'inference_times': self.inference_times,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'backbone': self.config.BACKBONE,
                'num_classes': self.config.NUM_CLASSES,
                'learning_rate': self.config.LEARNING_RATE,
                'batch_size': self.config.BATCH_SIZE,
                'num_epochs': self.config.NUM_EPOCHS,
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics saved to {filename}")
    
    def load_metrics(self, filename):
        
        with open(filename, 'r') as f:
            metrics = json.load(f)
        
        self.train_losses = metrics['train_losses']
        self.val_losses = metrics['val_losses']
        self.train_batch_losses = metrics['train_batch_losses']
        self.learning_rates = metrics['learning_rates']
        self.mAP_values = metrics['mAP_values']
        self.class_APs = defaultdict(list, metrics['class_APs'])
        self.precisions = defaultdict(list, metrics['precisions'])
        self.recalls = defaultdict(list, metrics['recalls'])
        self.f1_scores = defaultdict(list, metrics['f1_scores'])
        self.confusion_matrices = metrics['confusion_matrices']
        self.inference_times = metrics['inference_times']
        
        print(f"Metrics loaded from {filename}")
    
    def plot_losses(self, save_path=None):
      
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        if self.train_losses:
            epochs, losses = zip(*self.train_losses)
            plt.plot(epochs, losses, 'b-', label='Training Loss')
        
        # Plot validation loss
        if self.val_losses:
            epochs, losses = zip(*self.val_losses)
            plt.plot(epochs, losses, 'r-', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Loss plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_batch_losses(self, save_path=None):
        
        if not self.train_batch_losses:
            print("No batch losses to plot.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Extract batch indices and losses
        epochs, batch_indices, losses = zip(*self.train_batch_losses)
        
        # Calculate global batch index
        global_batch_indices = [e * len(set(batch_indices)) + b for e, b in zip(epochs, batch_indices)]
        
        plt.plot(global_batch_indices, losses, 'b-')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Batch Losses')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Batch loss plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_learning_rate(self, save_path=None):
        
        if not self.learning_rates:
            print("No learning rates to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Extract epochs and learning rates
        epochs, learning_rates = zip(*self.learning_rates)
        
        plt.plot(epochs, learning_rates, 'g-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Learning rate plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_mAP(self, save_path=None):
      
        if not self.mAP_values:
            print("No mAP values to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Extract epochs and mAP values
        epochs, mAP_values = zip(*self.mAP_values)
        
        plt.plot(epochs, mAP_values, 'm-')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('Mean Average Precision')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"mAP plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_class_APs(self, save_path=None):
        
        if not self.class_APs:
            print("No class APs to plot.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot AP for each class
        for class_name, ap_values in self.class_APs.items():
            if ap_values:
                epochs, ap = zip(*ap_values)
                plt.plot(epochs, ap, label=class_name)
        
        plt.xlabel('Epoch')
        plt.ylabel('AP')
        plt.title('Class Average Precisions')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Class AP plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_precision_recall_f1(self, class_name=None, save_path=None):
      
        if class_name and class_name not in self.precisions:
            print(f"No metrics frf class {class_name}.")
            return
        
        if not class_name and not self.precisions:
            print("No precision/recall/F1 metrics to plot.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot metrics for the specified class or all classes
        classes_to_plot = [class_name] if class_name else self.precisions.keys()
        
        for cls in classes_to_plot:
            if cls in self.precisions and self.precisions[cls]:
                epochs, precision = zip(*self.precisions[cls])
                plt.plot(epochs, precision, label=f'{cls} Precision')
            
            if cls in self.recalls and self.recalls[cls]:
                epochs, recall = zip(*self.recalls[cls])
                plt.plot(epochs, recall, label=f'{cls} Recall')
            
            if cls in self.f1_scores and self.f1_scores[cls]:
                epochs, f1 = zip(*self.f1_scores[cls])
                plt.plot(epochs, f1, label=f'{cls} F1')
        
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision, Recall, and F1 Scores')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Precision/recall/F1 plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_confusion_matrix(self, epoch=-1, save_path=None):
     
        if not self.confusion_matrices:
            print("No confusion matrices to plot.")
            return
        
        # Get confusion matrix for the specified epoch
        if epoch == -1:
            epoch, cm = self.confusion_matrices[-1]
        else:
            for e, matrix in self.confusion_matrices:
                if e == epoch:
                    cm = matrix
                    break
            else:
                print(f"No confusion matrix for epoch {epoch}.")
                return
        
        cm = np.array(cm)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names[1:],  # Skip background class
            yticklabels=self.class_names[1:]   # Skip background class
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Epoch {epoch})')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_inference_time(self, save_path=None):
       
        if not self.inference_times:
            print("No inference times to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        
        epochs, times = zip(*self.inference_times)
        
        plt.plot(epochs, times, 'c-')
        plt.xlabel('Epoch')
        plt.ylabel('Inference Time (ms)')
        plt.title('Model Inference Time')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Inference time plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_all_metrics(self, save_dir=None):
      
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
 
        save_path = os.path.join(save_dir, 'losses.png') if save_dir else None
        self.plot_losses(save_path)
        
      
        save_path = os.path.join(save_dir, 'batch_losses.png') if save_dir else None
        self.plot_batch_losses(save_path)
        

        save_path = os.path.join(save_dir, 'learning_rate.png') if save_dir else None
        self.plot_learning_rate(save_path)
        
        
        save_path = os.path.join(save_dir, 'mAP.png') if save_dir else None
        self.plot_mAP(save_path)
        
        save_path = os.path.join(save_dir, 'class_APs.png') if save_dir else None
        self.plot_class_APs(save_path)
        
        save_path = os.path.join(save_dir, 'precision_recall_f1.png') if save_dir else None
        self.plot_precision_recall_f1(save_path=save_path)
        
        save_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
        self.plot_confusion_matrix(save_path=save_path)
        
        save_path = os.path.join(save_dir, 'inference_time.png') if save_dir else None
        self.plot_inference_time(save_path)
    
    def get_latest_metrics(self):
        
        metrics = {}
        
        # Get latest train loss
        if self.train_losses:
            metrics['train_loss'] = self.train_losses[-1][1]
        
        # Get latest val loss
        if self.val_losses:
            metrics['val_loss'] = self.val_losses[-1][1]
        
        # Get latest mAP
        if self.mAP_values:
            metrics['mAP'] = self.mAP_values[-1][1]
        
        # Get latest class APs
        metrics['class_APs'] = {}
        for class_name, ap_values in self.class_APs.items():
            if ap_values:
                metrics['class_APs'][class_name] = ap_values[-1][1]
        
        # Get latest precision, recall, and F1 score
        metrics['precision'] = {}
        metrics['recall'] = {}
        metrics['f1_score'] = {}
        
        for class_name in self.precisions.keys():
            if self.precisions[class_name]:
                metrics['precision'][class_name] = self.precisions[class_name][-1][1]
            
            if self.recalls[class_name]:
                metrics['recall'][class_name] = self.recalls[class_name][-1][1]
            
            if self.f1_scores[class_name]:
                metrics['f1_score'][class_name] = self.f1_scores[class_name][-1][1]
        
        # Get latest inference time
        if self.inference_times:
            metrics['inference_time'] = self.inference_times[-1][1]
        
        return metrics
    
    def print_latest_metrics(self):
        
        metrics = self.get_latest_metrics()
        
        print("=" * 50)
        print("Latest Metrics")
        print("=" * 50)
        
        if 'train_loss' in metrics:
            print(f"Training Loss: {metrics['train_loss']:.4f}")
        
        if 'val_loss' in metrics:
            print(f"Validation Loss: {metrics['val_loss']:.4f}")
        
        if 'mAP' in metrics:
            print(f"mAP: {metrics['mAP']:.4f}")
        
        if 'class_APs' in metrics:
            print("\nClass APs:")
            for class_name, ap in metrics['class_APs'].items():
                print(f"  {class_name}: {ap:.4f}")
        
        if 'precision' in metrics and 'recall' in metrics and 'f1_score' in metrics:
            print("\nPrecision, Recall, and F1 Score:")
            for class_name in metrics['precision'].keys():
                print(f"  {class_name}:")
                print(f"    Precision: {metrics['precision'].get(class_name, 'N/A'):.4f}")
                print(f"    Recall: {metrics['recall'].get(class_name, 'N/A'):.4f}")
                print(f"    F1 Score: {metrics['f1_score'].get(class_name, 'N/A'):.4f}")
        
        if 'inference_time' in metrics:
            print(f"\nInference Time: {metrics['inference_time']:.2f} ms")
        
        print("=" * 50)


def compute_iou(box1, box2):
# maths is wrong double check

    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Compute area of intersection
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    
    # Compute area of union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Compute IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


def compute_ap(precision, recall):

    # Sort by recall
    indices = np.argsort(recall)
    recall = np.array(recall)[indices]
    precision = np.array(precision)[indices]
    
    # Compute AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap


def compute_map(predictions, targets, iou_threshold=0.5):

    # Initialize variables
    class_predictions = defaultdict(list)
    class_targets = defaultdict(list)
    
    # Group predictions and targets by class
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        
        target_boxes = target['boxes']
        target_labels = target['labels']
        
        # Group predictions by class
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            class_predictions[label.item()].append({
                'box': box,
                'score': score.item(),
                'matched': False
            })
        
        # Group targets by class
        for box, label in zip(target_boxes, target_labels):
            class_targets[label.item()].append({
                'box': box,
                'matched': False
            })
    
    # Compute AP for each class
    class_aps = {}
    
    for class_id in class_targets.keys():
        # Sort predictions by score
        class_predictions[class_id].sort(key=lambda x: x['score'], reverse=True)
        
        # Initialize precision and recall arrays
        precision = []
        recall = []
        
        # Initialize counters
        tp = 0
        fp = 0
        
        # Total number of ground truth objects
        n_gt = len(class_targets[class_id])
        
        # Process each prediction
        for pred in class_predictions[class_id]:
            # Find the best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(class_targets[class_id]):
                if gt['matched']:
                    continue
                
                iou = compute_iou(pred['box'], gt['box'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if the prediction matches a ground truth
            if best_iou >= iou_threshold:
                tp += 1
                class_targets[class_id][best_gt_idx]['matched'] = True
                pred['matched'] = True
            else:
                fp += 1
            
            # Compute precision and recall
            precision.append(tp / (tp + fp))
            recall.append(tp / n_gt if n_gt > 0 else 0)
        
        # Compute AP
        if precision and recall:
            ap = compute_ap(precision, recall)
        else:
            ap = 0
        
        class_aps[class_id] = ap
    
    # Compute mAP
    if class_aps:
        mAP = sum(class_aps.values()) / len(class_aps)
    else:
        mAP = 0
    
    return mAP, class_aps


def compute_confusion_matrix(predictions, targets, num_classes):

    cm = np.zeros((num_classes - 1, num_classes - 1), dtype=np.int32)  # Exclude background class
    
    # Process each image
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        
        target_boxes = target['boxes']
        target_labels = target['labels']
        
        # Match predictions to targets
        for target_idx, (target_box, target_label) in enumerate(zip(target_boxes, target_labels)):
            # Skip background class
            if target_label == 0:
                continue
            
            # Find the best matching prediction
            best_iou = 0
            best_pred_idx = -1
            
            for pred_idx, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                # Skip background class
                if pred_label == 0:
                    continue
                
                iou = compute_iou(target_box, pred_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            
            # Update confusion matrix
            if best_pred_idx != -1 and best_iou >= 0.5:
                pred_label = pred_labels[best_pred_idx].item()
                cm[target_label.item() - 1, pred_label - 1] += 1
            else:
                # No matching prediction, count as background prediction
                pass
    
    return cm


def compute_precision_recall_f1(predictions, targets, class_id, iou_threshold=0.5):
    """
    Compute precision, recall, and F1 score for a specific class.
    
    Args:
        predictions (list): List of prediction dictionaries.
            Each dictionary should have 'boxes', 'labels', and 'scores' keys.
        targets (list): List of target dictionaries.
            Each dictionary should have 'boxes' and 'labels' keys.
        class_id (int): Class ID to compute metrics for.
        iou_threshold (float): IoU threshold for considering a detection as correct.
        
    Returns:
        tuple: Tuple containing precision, recall, and F1 score.
    """
    # Initialize counters
    tp = 0
    fp = 0
    fn = 0
    
    # Process each image
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        
        target_boxes = target['boxes']
        target_labels = target['labels']
        
        # Get predictions for the specified class
        class_pred_indices = [i for i, label in enumerate(pred_labels) if label.item() == class_id]
        class_pred_boxes = [pred_boxes[i] for i in class_pred_indices]
        class_pred_scores = [pred_scores[i] for i in class_pred_indices]
        
        # Get targets for the specified class
        class_target_indices = [i for i, label in enumerate(target_labels) if label.item() == class_id]
        class_target_boxes = [target_boxes[i] for i in class_target_indices]
        
        # Mark all targets as unmatched
        target_matched = [False] * len(class_target_boxes)
        
        # Sort predictions by score
        sorted_indices = sorted(range(len(class_pred_scores)), key=lambda i: class_pred_scores[i], reverse=True)
        
        # Process each prediction
        for i in sorted_indices:
            pred_box = class_pred_boxes[i]
            
            # Find the best matching target
            best_iou = 0
            best_target_idx = -1
            
            for j, target_box in enumerate(class_target_boxes):
                if target_matched[j]:
                    continue
                
                iou = compute_iou(pred_box, target_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j
            
            # Check if the prediction matches a target
            if best_target_idx != -1 and best_iou >= iou_threshold:
                tp += 1
                target_matched[best_target_idx] = True
            else:
                fp += 1
        
        # Count unmatched targets as false negatives
        fn += sum(1 for matched in target_matched if not matched)
    
    # Compute precision, recall, and F1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1


def compute_metrics_for_all_classes(predictions, targets, class_names, iou_threshold=0.5):
    """
    Compute metrics for all classes.
    
    Args:
        predictions (list): List of prediction dictionaries.
            Each dictionary should have 'boxes', 'labels', and 'scores' keys.
        targets (list): List of target dictionaries.
            Each dictionary should have 'boxes' and 'labels' keys.
        class_names (list): List of class names.
        iou_threshold (float): IoU threshold for considering a detection as correct.
        
    Returns:
        dict: Dictionary containing metrics for all classes.
    """
    # Compute mAP and class APs
    mAP, class_aps = compute_map(predictions, targets, iou_threshold)
    
    # Compute precision, recall, and F1 score for each class
    metrics = {
        'mAP': mAP,
        'class_APs': {},
        'precision': {},
        'recall': {},
        'f1_score': {}
    }
    
    for class_id in range(1, len(class_names)):  # Skip background class
        # Get class name
        class_name = class_names[class_id]
        
        # Get class AP
        metrics['class_APs'][class_name] = class_aps.get(class_id, 0)
        
        # Compute precision, recall, and F1 score
        precision, recall, f1 = compute_precision_recall_f1(predictions, targets, class_id, iou_threshold)
        
        metrics['precision'][class_name] = precision
        metrics['recall'][class_name] = recall
        metrics['f1_score'][class_name] = f1
    
    return metrics
