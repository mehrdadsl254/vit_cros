"""
Complete test script for cross-image similarity computation.
Tests similarity between two images with rearranged car objects.
"""
import os
import sys
import torch
import numpy as np
import cv2
import json
from pathlib import Path


# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.create_test_data import create_test_data
from utils.dataset import ADE20KSegmentation, FeatureExtractor
from trainer_ce import TrainerCE
from utils.utils import set_random_seed
import yaml
from omegaconf import OmegaConf

def create_simple_test_images(output_dir):
    """Create simple test images with 4 colored rectangles representing cars"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create image 1: A (red), B (green), C (blue), D (yellow)
    img1 = np.zeros((512, 512, 3), dtype=np.uint8)
    h, w = img1.shape[:2]
    mid_h, mid_w = h // 2, w // 2
    
    # Top-left: Red (A)
    img1[0:mid_h, 0:mid_w] = [255, 0, 0]
    # Top-right: Green (B)
    img1[0:mid_h, mid_w:w] = [0, 255, 0]
    # Bottom-left: Blue (C)
    img1[mid_h:h, 0:mid_w] = [0, 0, 255]
    # Bottom-right: Yellow (D)
    img1[mid_h:h, mid_w:w] = [255, 255, 0]
    
    # Create image 2: D (yellow), C (blue), B (green), A (red)
    img2 = np.zeros((512, 512, 3), dtype=np.uint8)
    # Top-left: Yellow (D)
    img2[0:mid_h, 0:mid_w] = [255, 255, 0]
    # Top-right: Blue (C)
    img2[0:mid_h, mid_w:w] = [0, 0, 255]
    # Bottom-left: Green (B)
    img2[mid_h:h, 0:mid_w] = [0, 255, 0]
    # Bottom-right: Red (A)
    img2[mid_h:h, mid_w:w] = [255, 0, 0]
    
    img1_path = os.path.join(output_dir, "image1.png")
    img2_path = os.path.join(output_dir, "image2.png")
    
    cv2.imwrite(img1_path, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(img2_path, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    
    return img1_path, img2_path

def test_cross_image_similarity(cfg_path=None, checkpoint_path=None):
    """
    Main test function for cross-image similarity.
    """
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
    
    # Create test images if they don't exist
    workspace_root = Path(__file__).parent.parent
    img1_path = workspace_root / "colorf1.png"
    img2_path = workspace_root / "colorf2.png"
    
    if not img1_path.exists() or not img2_path.exists():
        print("Creating test images...")
        create_simple_test_images(workspace_root)
    
    # Create test data in ADE20K format
    data_dir = workspace_root / "data"
    print(f"Creating test data in ADE20K format...")
    test_folder, instance_mask1, instance_mask2, class_mask1, class_mask2 = create_test_data(
        str(img1_path), 
        str(img2_path), 
        str(data_dir),
        base_name="test_car_rearrangement"
    )
    
    # Initialize trainer (this will load the model)
    output_dir = os.path.join(data_dir, "outputs", "test_cross_image")
    os.makedirs(output_dir, exist_ok=True)
    
    # Update config for test
    cfg.mode = 'eval_all'
    cfg.dataset.data_dir = str(data_dir)
    
    print("Initializing trainer and loading model...")
    trainer = TrainerCE(cfg, output_dir)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=True)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run cross-image similarity test
    print("\n" + "="*50)
    print("Running Cross-Image Similarity Test")
    print("="*50)

    similarity_matrix, mapping_info = trainer.test_cross_image_similarity(
        cfg,
        image1_path=str(img1_path),
        image2_path=str(img2_path),
        masks=[class_mask1, class_mask2],
        bboxes=[instance_mask1, instance_mask2],
    )
    
    if similarity_matrix is not None:
        print("\n" + "="*50)
        print("Results Summary")
        print("="*50)
        
        # Check if diagonal has highest values (correct matching)
        similarity_np = similarity_matrix.cpu().numpy()
        diagonal_values = np.diag(similarity_np)
        
        # Get max off-diagonal value
        mask = ~np.eye(4, dtype=bool)
        off_diagonal_max = np.max(similarity_np[mask]) if mask.sum() > 0 else 0
        
        print(f"\nDiagonal (correct matches) values: {diagonal_values}")
        print(f"Max off-diagonal value: {off_diagonal_max}")
        
        # Check if diagonal is higher than off-diagonal
        if np.all(diagonal_values > off_diagonal_max):
            print("\n✓ SUCCESS: All correct matches have higher similarity than incorrect matches!")
        else:
            print("\n✗ WARNING: Some incorrect matches have higher similarity than correct ones.")
        
        # Save results
        results_path = os.path.join(output_dir, "cross_image_similarity_results.npz")
        np.savez(
            results_path,
            similarity_matrix=similarity_np,
            diagonal_values=diagonal_values,
            mapping_info=mapping_info
        )
        print(f"\nResults saved to: {results_path}")
        
        return similarity_matrix, mapping_info
    else:
        print("Error: Could not compute similarity matrix")
        return None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test cross-image similarity")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    test_cross_image_similarity(
        cfg_path=args.config,
        checkpoint_path=args.checkpoint
    )

