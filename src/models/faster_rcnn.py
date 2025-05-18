
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet18, resnet34, resnet50, resnet101


def create_faster_rcnn_model(config):
   
   
    # Get the number of classes
    num_classes = config.NUM_CLASSES
    
    # Get the backbone
    backbone_name = config.BACKBONE
    pretrained = config.PRETRAINED
    trainable_backbone_layers = config.TRAINABLE_BACKBONE_LAYERS
    
    # Create the backbone
    if backbone_name == 'resnet18':
        backbone = resnet18(pretrained=pretrained)
    elif backbone_name == 'resnet34':
        backbone = resnet34(pretrained=pretrained)
    elif backbone_name == 'resnet50':
        backbone = resnet50(pretrained=pretrained)
    elif backbone_name == 'resnet101':
        backbone = resnet101(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    # Freeze layers
    if trainable_backbone_layers < 5:
        for name, parameter in backbone.named_parameters():
            if name.startswith('layer4') and trainable_backbone_layers >= 4:
                parameter.requires_grad_(True)
            elif name.startswith('layer3') and trainable_backbone_layers >= 3:
                parameter.requires_grad_(True)
            elif name.startswith('layer2') and trainable_backbone_layers >= 2:
                parameter.requires_grad_(True)
            elif name.startswith('layer1') and trainable_backbone_layers >= 1:
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)
    
    # Remove the last two layers (avgpool and fc)
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    
    # RCNN needs to know  number of output channels in the backbone
    backbone.out_channels = 512 if backbone_name in ['resnet18', 'resnet34'] else 2048
    
    # Create anchor generator
    anchor_sizes = config.RPN_ANCHOR_SIZES
    aspect_ratios = config.RPN_ANCHOR_RATIOS
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    
    # Create RoI pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    # hardcide for CFG must be abtter way //cehck
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=config.MIN_SIZE,
        max_size=config.MAX_SIZE,
        image_mean=config.IMAGE_MEAN,
        image_std=config.IMAGE_STD,
        rpn_pre_nms_top_n_train=config.RPN_PRE_NMS_TOP_N_TRAIN,
        rpn_pre_nms_top_n_test=config.RPN_PRE_NMS_TOP_N_TEST,
        rpn_post_nms_top_n_train=config.RPN_POST_NMS_TOP_N_TRAIN,
        rpn_post_nms_top_n_test=config.RPN_POST_NMS_TOP_N_TEST,
        rpn_nms_thresh=config.RPN_NMS_THRESH,
        rpn_fg_iou_thresh=config.RPN_FG_IOU_THRESH,
        rpn_bg_iou_thresh=config.RPN_BG_IOU_THRESH,
        rpn_batch_size_per_image=config.RPN_BATCH_SIZE_PER_IMAGE,
        rpn_positive_fraction=config.RPN_POSITIVE_FRACTION,
        box_score_thresh=config.BOX_SCORE_THRESH,
        box_nms_thresh=config.BOX_NMS_THRESH,
        box_detections_per_img=config.BOX_DETECTIONS_PER_IMG,
        box_fg_iou_thresh=config.BOX_FG_IOU_THRESH,
        box_bg_iou_thresh=config.BOX_BG_IOU_THRESH,
        box_batch_size_per_image=config.BOX_BATCH_SIZE_PER_IMAGE,
        box_positive_fraction=config.BOX_POSITIVE_FRACTION
    )
    
    return model


def create_model_for_ablation(config, backbone_name=None, anchor_sizes=None, anchor_ratios=None):

    # FALLBACK incase it failes it wont tho...
    backbone_name = backbone_name or config.BACKBONE
    anchor_sizes = anchor_sizes or config.RPN_ANCHOR_SIZES
    anchor_ratios = anchor_ratios or config.RPN_ANCHOR_RATIOS
    
    # HAVE TO PASS as a copy
    class CustomConfig:
        pass
    
    custom_config = CustomConfig()
    for key, value in vars(config).items():
        setattr(custom_config, key, value)
    
 
    custom_config.BACKBONE = backbone_name
    custom_config.RPN_ANCHOR_SIZES = anchor_sizes
    custom_config.RPN_ANCHOR_RATIOS = anchor_ratios

    return create_faster_rcnn_model(custom_config)
