
import os
import torch
import time
import json
from datetime import datetime


def save_model(model, optimizer, epoch, batch_idx, loss, config, is_best=False):

    # Create checkpoint directory if it doesn't exist
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    
    # Create checkpoint filename
    if is_best:
        checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, 'model_best.pth')
    else:
        checkpoint_path = os.path.join(
            config.CHECKPOINTS_DIR, 
            f'model_epoch_{epoch}_batch_{batch_idx}.pth'
        )
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }, checkpoint_path)
    
    # Save metadata
    metadata_path = os.path.join(
        config.CHECKPOINTS_DIR, 
        f'metadata_epoch_{epoch}_batch_{batch_idx}.json'
    )
    
    with open(metadata_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'loss': float(loss),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_best': is_best,
            'backbone': config.BACKBONE,
            'num_classes': config.NUM_CLASSES,
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
        }, f, indent=4)
    
    print(f"Model saved to {checkpoint_path}")
    print(f"Metadata saved to {metadata_path}")


def load_model(model, optimizer, checkpoint_path):# if error here change to cuda.availabvle
 
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # Load checkpoint
     #  checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get epoch and batch index
    epoch = checkpoint.get('epoch', 0)
    batch_idx = checkpoint.get('batch_idx', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss:.4f}")
    
    return model, optimizer, epoch, batch_idx, loss


def get_latest_checkpoint(checkpoints_dir):
   
    # Check if checkpoints directory exists
    if not os.path.exists(checkpoints_dir):
        return None
    
    # Get all checkpoint files
    checkpoint_files = [
        os.path.join(checkpoints_dir, f) 
        for f in os.listdir(checkpoints_dir) 
        if f.endswith('.pth') and f != 'model_best.pth'
    ]
    
    # Return None if no checkpoints found
    if not checkpoint_files:
        return None
    
    # Get the latest checkpoint based on modification time
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    
    return latest_checkpoint


def get_best_checkpoint(checkpoints_dir):
  
    # Check if best checkpoint exists
    best_checkpoint_path = os.path.join(checkpoints_dir, 'model_best.pth')
    
    if os.path.exists(best_checkpoint_path):
        return best_checkpoint_path
    else:
        return None


def get_model_size(model):
 
    # Get model state dict
    state_dict = model.state_dict()
    
    # Calculate size in bytes
    size_bytes = sum(param.nelement() * param.element_size() for param in state_dict.values())
    
    # Convert to MB
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb


def get_inference_time(model, input_tensor, num_runs=100, warmup_runs=10):
 
    # Set model to evaluation mode
    model.eval()
    
    # Move input to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    end_time = time.time()
    
    # Calculate average inference time in milliseconds
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    
    return avg_time_ms


def count_parameters(model):
  
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model, input_size=(3, 800, 800)):
  
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size
    size_mb = get_model_size(model)
    
    # Create summary
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': size_mb,
        'input_size': input_size,
    }
    
    return summary


def print_model_summary(model, input_size=(3, 800, 800)):
   
    # Get model summary
    summary = get_model_summary(model, input_size)
    
    # Print summary
    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(f"Input Size: {summary['input_size']}")
    print(f"Total Parameters: {summary['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"Non-Trainable Parameters: {summary['non_trainable_parameters']:,}")
    print(f"Model Size: {summary['model_size_mb']:.2f} MB")
    print("=" * 50)
