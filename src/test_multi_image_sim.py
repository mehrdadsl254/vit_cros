"""
Test script for multi-image similarity computation.
Tests similarity between 6 images, each containing a single object.
"""
import os
import sys
import torch
import numpy as np
import cv2
import yaml
from pathlib import Path
from omegaconf import OmegaConf

import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.create_test_data import segment_single_object, load_sam_model
# from utils.dataset import TrainerCE
from utils.utils import set_random_seed

# Note: TrainerCE is in src/trainer_ce.py, importing from there
from trainer_ce import TrainerCE

def create_composite_overview(image_paths, instance_masks, image_names, output_dir):
    """Create a single composite image showing all images, their names, and masks."""
    num_images = len(image_paths)
    print("Creating composite overview...")
    
    # Setup figure: N rows, 2 cols (Image, Mask)
    # Adjust height based on number of images
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 3 * num_images))
    if num_images == 1:
        axes = np.array([axes])
        
    fig.suptitle("Segmentation Overview", fontsize=16)
    
    for i in range(num_images):
        # Load image (RGB)
        img = cv2.imread(image_paths[i])[:, :, ::-1]
        mask = (instance_masks[i] > 0).astype(np.uint8)
        
        # Plot Original Image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{image_names[i]}\n(Original)", fontsize=12)
        axes[i, 0].axis('off')
        
        # Plot Mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"{image_names[i]}\n(Segmentation Mask)", fontsize=12)
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    # Adjust subplot top to make room for suptitle
    plt.subplots_adjust(top=0.95)
    
    save_path = os.path.join(output_dir, "all_images_overview.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Composite overview saved to {save_path}")

def test_multi_image_similarity(image_paths, image_names=None, cfg_path=None, checkpoint_path=None, output_dir=None):
    """
    Main test function for multi-image similarity with 6 single-object images.
    """
    if len(image_paths) != 6:
        print(f"Warning: Expected 6 images, got {len(image_paths)}")
        
    if image_names is None:
        image_names = [f"Img {i+1}" for i in range(len(image_paths))]
        
    if len(image_names) != len(image_paths):
        print(f"Warning: Number of names ({len(image_names)}) does not match number of images ({len(image_paths)}). Using defaults.")
        image_names = [f"Img {i+1}" for i in range(len(image_paths))]
        
    # Load config
    if cfg_path is None:
        cfg_path = "cfgs/config.yaml"
        
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    cfg = OmegaConf.create(cfg_dict)
    
    # Set device
    if not torch.cuda.is_available():
        cfg.trainer.device = 'cpu'
        cfg.device = 'cpu'
        
    set_random_seed(cfg.seed)
    
    # Initialize output directory first to save segmentation vis
    if output_dir is None:
        output_dir = "outputs/multi_image_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load SAM if available (for better segmentation)
    print("Loading SAM for segmentation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_predictor, sam_mask_generator = load_sam_model(device=device)
    
    # Process images
    masks = []
    bboxes = []
    
    print("Segmenting images...")
    for i, path in enumerate(image_paths):
        name = image_names[i]
        print(f"Processing {name} ({path})...")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
            
        # Segment single object
        instance_mask, class_mask = segment_single_object(
            img, 
            use_sam=True, 
            sam_predictor=sam_predictor,
            sam_mask_generator=sam_mask_generator
        )
        
        masks.append(class_mask)
        bboxes.append(instance_mask)
        
        # Save visualization of segmentation
        # 1. Save binary mask
        vis_mask = (instance_mask > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, f"segmentation_mask_{i}_{name}.png"), vis_mask)
        
        # 2. Save overlay on original image
        overlay = img.copy()
        green_mask = np.zeros_like(img)
        green_mask[instance_mask > 0] = [0, 255, 0] # BGR
        
        # Blend
        alpha = 0.5
        cv2.addWeighted(green_mask, alpha, overlay, 1 - alpha, 0, overlay)
        
        # Draw contour
        contours, _ = cv2.findContours((instance_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(output_dir, f"segmentation_overlay_{i}_{name}.png"), overlay)
        
        # 3. Save masked object
        masked_img = img.copy()
        masked_img[instance_mask == 0] = 0
        cv2.imwrite(os.path.join(output_dir, f"segmentation_object_{i}_{name}.png"), masked_img)
            
    # Create composite overview
    create_composite_overview(image_paths, bboxes, image_names, output_dir)
    
    cfg.mode = 'eval_all'
    print("Initializing trainer and loading model...")
    trainer = TrainerCE(cfg, output_dir)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=True)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
    # Run similarity test
    sim_matrix = trainer.test_multi_image_similarity(
        cfg, 
        image_paths=image_paths, 
        masks=masks, 
        bboxes=bboxes
    )
    
    # Save matrix raw data
    sim_np = sim_matrix.cpu().numpy()
    np.save(os.path.join(output_dir, "similarity_matrix_6x6.npy"), sim_np)
    
    # Plot Similarity Matrix
    print("Plotting similarity matrix...")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_np, cmap="viridis")
    
    # Add labels
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(image_names)
    ax.set_yticklabels(image_names)
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(6):
        for j in range(6):
            text = ax.text(j, i, f"{sim_np[i, j]:.2f}",
                           ha="center", va="center", color="w" if sim_np[i, j] < 0.5 else "black")
    
    ax.set_title("Cross-Image Object Similarity")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "similarity_matrix_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    
    print(f"Results saved to {output_dir}")
    print(f"- Matrix: {os.path.join(output_dir, 'similarity_matrix_6x6.npy')}")
    print(f"- Plot: {plot_path}")
    print(f"- Overview: {os.path.join(output_dir, 'all_images_overview.png')}")
    
    return sim_matrix

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs='+', required=True, help="List of 6 image paths")
    parser.add_argument("--names", nargs='+', help="List of 6 image names (optional)")
    parser.add_argument("--config", default="cfgs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="outputs/test_6_images")
    
    args = parser.parse_args()
    
    test_multi_image_similarity(
        args.images,
        image_names=args.names,
        cfg_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output
    )
