
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from datetime import datetime
import torchvision.transforms as T


def visualize_detection(image, boxes, labels, scores=None, class_names=None, threshold=0.5, save_path=None):

    # Convert to numpy arrays
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    # Create high-quality figure
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots(1, figsize=(12, 9), dpi=300)
    
    # Improve image quality
    if len(image_np.shape) == 2:  # If grayscale
        plt.imshow(image_np, cmap='gray')
    else:
        plt.imshow(image_np)
    
    # Generate random colors for each class
    if class_names is None:
        class_names = [str(i) for i in range(max(labels) + 1)]
    
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    # Draw bounding boxes and labels
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if scores is not None and scores[i] < threshold:
            continue
        
        x1, y1, x2, y2 = box
        class_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        color = colors[label % len(colors)]
        
        # Create rectangle with thicker lines
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3,  # Increased line width
            edgecolor=color,
            facecolor='none'
        )
        
        ax.add_patch(rect)
        
        # Add label with improved visibility
        if scores is not None:
            label_text = f"{class_name}: {scores[i]:.2f}"
        else:
            label_text = class_name
            
        ax.text(
            x1, y1 - 5,
            label_text,
            color='black',  # Black text
            fontsize=12,    # Larger font
            fontweight='bold',  # Bold text
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor=color,
                linewidth=2,
                pad=1
            )
        )
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    if save_path:
        plt.savefig(
            save_path,
            bbox_inches='tight',
            pad_inches=0.1,
            dpi=300,
            format='png',
            facecolor='white',
            edgecolor='none'
        )
        plt.close()
    
    # Convert figure to image array
    fig.canvas.draw()
    vis_image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    
    return vis_image


def visualize_batch(images, targets, predictions=None, class_names=None, threshold=0.5, max_images=16, save_dir=None):

    # Limit number of images
    num_images = min(len(images), max_images)
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Visualize each image
    vis_images = []
    
    for i in range(num_images):
        # Get image and target
        image = images[i]
        target = targets[i]
        
        # Improved image tensor conversion
        if isinstance(image, torch.Tensor):
            # Get mean and std from config if available
            try:
                from config.model_config import IMAGE_MEAN, IMAGE_STD
                mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
                std = torch.tensor(IMAGE_STD).view(3, 1, 1)
                image = image * std + mean  # Denormalize
            except ImportError:
                # Default denormalization
                image = image * 0.5 + 0.5
            
            # Convert to numpy with better quality
            image = image.cpu().permute(1, 2, 0).numpy()
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Get ground truth boxes and labels
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        
        # Visualize ground truth
        save_path = os.path.join(save_dir, f"image_{i}_gt.png") if save_dir else None
        gt_vis = visualize_detection(
            image, gt_boxes, gt_labels,
            class_names=class_names,
            save_path=save_path
        )
        vis_images.append(gt_vis)
        
        # Visualize predictions if available
        if predictions:
            pred = predictions[i]
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
            pred_scores = pred['scores']
            
            save_path = os.path.join(save_dir, f"image_{i}_pred.png") if save_dir else None
            pred_vis = visualize_detection(
                image, pred_boxes, pred_labels, pred_scores,
                class_names=class_names,
                threshold=threshold,
                save_path=save_path
            )
            vis_images.append(pred_vis)
    
    return vis_images


def visualize_predictions(model, images, targets, class_names, device, threshold=0.5, save_dir=None):
   
    # Set model to evaluation mode
    model.eval()
    
    # Move images to device
    images = [img.to(device) for img in images]
    
    # Get predictions
    with torch.no_grad():
        predictions = model(images)
    
    # Visualize batch
    vis_images = visualize_batch(
        images, targets, predictions,
        class_names=class_names,
        threshold=threshold,
        save_dir=save_dir
    )
    
    return vis_images



def visualize_feature_maps(model, image, layer_name, save_dir=None):

    # Set model to evaluation mode
    model.eval()
    
    # Register hook to get feature maps
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output)
    
    # Find the layer
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break
    else:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Forward pass
    with torch.no_grad():
        _ = model([image])
    
    # Remove hook
    hook.remove()
    
    # Get feature maps
    feature_map = feature_maps[0]
    
    # If feature map is a list, get the first element
    if isinstance(feature_map, list):
        feature_map = feature_map[0]
    
    # Move to CPU and convert to numpy
    feature_map = feature_map.cpu().numpy()
    
    # If feature map has batch dimension, remove it
    if len(feature_map.shape) == 4:
        feature_map = feature_map[0]
    
    # Get number of channels
    num_channels = feature_map.shape[0]
    
    # Compute grid size
    grid_size = int(np.ceil(np.sqrt(num_channels)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    
    # Plot each channel
    for i in range(grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        ax = axes[row, col]
        
        if i < num_channels:
            # Get channel
            channel = feature_map[i]
            
            # Normalize channel
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            
            # Plot channel
            ax.imshow(channel, cmap='viridis')
            ax.set_title(f"Channel {i}")
        
        # Remove axis
        ax.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f"feature_maps_{layer_name}_{timestamp}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Feature maps saved to {save_path}")
    else:
        plt.show()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    vis_image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    
    return vis_image


def visualize_attention(model, image, layer_name, save_dir=None):
   
    # Set model to evaluation mode
    model.eval()
    
    # Register hook to get attention maps
    attention_maps = []
    
    def hook_fn(module, input, output):
        
        if isinstance(output, tuple) and len(output) > 1:
            attention_maps.append(output[1])
        else:
            attention_maps.append(output)
    
    # Find the layer
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break
    else:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Forward pass
    with torch.no_grad():
        _ = model([image])
    
    # Remove hook
    hook.remove()
    
    # Get attention maps
    attention_map = attention_maps[0]
    
    # Move to CPU and convert to numpy
    attention_map = attention_map.cpu().numpy()
    
    # If attention map has batch dimension, remove it
    if len(attention_map.shape) == 4:
        attention_map = attention_map[0]
    
    # Get number of heads
    num_heads = attention_map.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))
    
    # Plot each head
    for i in range(num_heads):
        # Get head
        head = attention_map[i]
        
        # Normalize head
        head = (head - head.min()) / (head.max() - head.min() + 1e-8)
        
        # Plot head
        if num_heads > 1:
            ax = axes[i]
        else:
            ax = axes
        
        im = ax.imshow(head, cmap='viridis')
        ax.set_title(f"Head {i}")
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f"attention_maps_{layer_name}_{timestamp}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Attention maps saved to {save_path}")
    else:
        plt.show()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    vis_image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    
    return vis_image


def visualize_class_activation_map(model, image, class_idx, layer_name, save_path=None):

    # Set model to evaluation mode
    model.eval()
    
    # Register hooks
    feature_maps = []
    gradients = []
    
    def forward_hook(module, input, output):
        feature_maps.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Find the layer
    for name, module in model.named_modules():
        if name == layer_name:
            forward_handle = module.register_forward_hook(forward_hook)
            backward_handle = module.register_backward_hook(backward_hook)
            break
    else:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Forward pass
    model.zero_grad()
    output = model([image])
    
    # Get score for target class
    if isinstance(output, list):
        output = output[0]
    
    score = output['scores'][class_idx]
    
    # Backward pass
    score.backward()
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Get feature maps and gradients
    feature_map = feature_maps[0]
    gradient = gradients[0]
    
    # If feature map is a list, get the first element
    if isinstance(feature_map, list):
        feature_map = feature_map[0]
    
    # If gradient is a list, get the first element
    if isinstance(gradient, list):
        gradient = gradient[0]
    
    # Move to CPU and convert to numpy
    feature_map = feature_map.cpu().numpy()
    gradient = gradient.cpu().numpy()
    
    # If feature map has batch dimension, remove it
    if len(feature_map.shape) == 4:
        feature_map = feature_map[0]
    
    # If gradient has batch dimension, remove it
    if len(gradient.shape) == 4:
        gradient = gradient[0]
    
    # Compute weights
    weights = np.mean(gradient, axis=(1, 2))
    
    # Compute CAM
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * feature_map[i]
    
    # ReLU
    cam = np.maximum(cam, 0)
    
    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Resize to image size
    cam = cv2.resize(cam, (image.shape[2], image.shape[1]))
    
    # Convert image to numpy array
    image_np = image.cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Combine image and heatmap
    result = heatmap * 0.3 + image_np * 0.7
    result = result.astype(np.uint8)
    
    # Create figure
    plt.figure(figsize=(12, 4))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Class Activation Map')
    plt.axis('off')
    
    # Plot combined image
    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.title('Combined')
    plt.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"CAM saved to {save_path}")
    else:
        plt.show()
    
    # Convert figure to numpy array
    plt.gcf().canvas.draw()
    vis_image = np.array(plt.gcf().canvas.renderer.buffer_rgba())
    plt.close()
    
    return vis_image
