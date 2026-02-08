import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Disable xformers to enforce CPU compatibility and avoid memory_efficient_attention error
os.environ["XFORMERS_DISABLED"] = "1"
import cv2
import pandas as pd
from tqdm import tqdm
import math
from scipy.stats import linregress

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../libs/dinov2"))
sys.path.insert(0, os.path.dirname(__file__)) # Add src to path

from trainer_ce import TrainerCE
from utils.dataset import ADE20KSegmentation, activations, inference_segmentor, render_segmentation, forward_transform
from utils.score import compute_batch_pairwise_similarity

def get_image_data(cfg, image_path):
    """
    Finds the image and its mask in the ADE20K dataset given the image path.
    """
    # Check if data_dir exists, if not try to find it
    if not os.path.exists(cfg.dataset.data_dir):
        # specific hardcoded guess based on findings
        potential_paths = [
            "/Users/mehrdadsalehi/Anki/doesp/vit-object-binding/vit-object-binding/data",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"),
            os.path.abspath("../data")
        ]
        for p in potential_paths:
            if os.path.exists(p):
                print(f"Data directory found at: {p}")
                cfg.dataset.data_dir = p
                # Update cfg object in dataset? No, we pass root to constructor.
                break

    dataset = ADE20KSegmentation(root=cfg.dataset.data_dir, image_set="val") # Search in val first, or iterate all
    # Just in case, check train as well if not found?
    # Or just search in the index file directly if accessible.


    
    # Let's search in the dataset's index
    filename = os.path.basename(image_path)
    found_idx = -1
    
    # We need to match the filename in the dataset index
    # dataset.index_ade20k['filename'] is a list of filenames
    
    # Try to match full path or just basename
    # The dataset uses relative paths in 'filename' list usually?
    # dataset.index_ade20k['filename'] contains like "ADE_val_00000001.jpg"
    
    print("Searching for image in ADE20K index...")
    for idx, fname in enumerate(dataset.index_ade20k['filename']):
        if fname == filename:
            found_idx = idx
            break
    print(f"Search in val finished. Found: {found_idx != -1}")
    
    if found_idx == -1:
        # Try train set
        dataset_train = ADE20KSegmentation(root=cfg.dataset.data_dir, image_set="train")
        for idx, fname in enumerate(dataset_train.index_ade20k['filename']):
            if fname == filename:
                dataset = dataset_train
                found_idx = idx
                break

    if found_idx == -1:
        raise ValueError(f"Image {filename} not found in ADE20K dataset index.")

    print(f"Found image at index {found_idx} in dataset index. Folder: {dataset.index_ade20k['folder'][found_idx]}")
    
    # Load directly using path from index, bypassing dataset.__getitem__ logic which expects mapped index
    image_id = found_idx
    full_file_name = os.path.join(dataset.index_ade20k['folder'][image_id], dataset.index_ade20k['filename'][image_id])
    
    # Check if dataset has utils_ade20k available in scope or needs import
    from utils.dataset import utils_ade20k
    
    info = utils_ade20k.loadAde20K(os.path.join(dataset.root, full_file_name))
    
    # Load Image and Segmentation Mask
    img = cv2.imread(info['img_name'])[:, :, ::-1]  # Convert BGR to RGB
    instance_mask = info['instance_mask'] # [768, 1024], 0: background, 1-n: object 
    seg = info['class_mask']
    
    return img, seg, instance_mask

@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg: DictConfig):
    if "image_path" not in cfg:
        print("Please provide image_path argument. Usage: python src/experiment_distance.py image_path=/path/to/image.jpg")
        return

    image_path = cfg.image_path
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing image: {image_path}")

    # Initialize Trainer to setup model and probe
    print("Initializing TrainerCE...")
    trainer = TrainerCE(cfg, output_dir)
    print("TrainerCE initialized.")
    feature_extractor = trainer.feature_extractor
    model = trainer.model
    device = trainer.device

    # Load image and mask
    print(f"Loading image data for {image_path}...")
    try:
        img, seg, instance_mask = get_image_data(cfg, image_path)
    except Exception as e:
        print(f"Error loading image data: {e}. Attempting to load directly if file exists and mask can be inferred.")
        # Fallback: if user provides a path that is not in index but exists locally
        if os.path.exists(image_path):
            img = cv2.imread(image_path)[:, :, ::-1]
            # Try to find mask file with standard naming? 
            # If not possible, we fail specific to the "object binding" requirement
            # But the user said "is from the dataset", so assumed it works.
            # If failed, just raise.
            raise e
    
    # Prepare inputs for model
    # FeatureExtractor expects batch
    # But we want to manually drive it to get all patches
    
    # 1. Inference to get activations
    # We can use inference_segmentor logic but just run backbone
    # feature_extractor.backbone_model is the DINOv2
    
    img_tensor = torch.from_numpy(img.copy()).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Normalize? DINOv2 usually expects specific normalization
    # dataset.py doesn't seem to have explicit normalization in __getitem__?
    # It seems inference_segmentor handles it? 
    # feature_extractor combines them.
    
    # Let's use feature_extractor.model (Segmentor) which has the backbone 
    # and the hook is registered on backbone.
    
    activations[str(cfg.model.num_layer[0])] = [] # Reset buffer
    
    with torch.no_grad():
        # Just running specific parts to trigger hooks
        # inference_segmentor(feature_extractor.model, img)
        # But inference_segmentor expects numpy image
        
        # Let's verify how inference_segmentor works. It is imported from dataset -> segment
        # It likely does the preprocessing.
        
        _ = inference_segmentor(feature_extractor.model, img)
        
    activation = activations[str(cfg.model.num_layer[0])]
    if len(activation) == 0:
        print("No activations captured.")
        return

    # activation is list of tensors. Usually one per window if sliding window or one if global.
    # Take the first one for simplicity or iterate? 
    # If the image is large, it might be sliding window.
    # But let's assume one pass for now or handle the list.
    
    # "patches from the image that is given to the model"
    # We will aggregate all patches from all windows?
    # Or just assume the image fits in one crop/resize?
    # dataset.py line 180 checks len(activation) != 2?
    
    # Let's take the first activation map
    act = activation[0] # [1, N_patches + 1, C] (including CLS?)
    # DINOv2 usually: [B, N, C]
    
    # Check shape
    print(f"Activation shape: {act.shape}")
    
    # Remove CLS token (usually index 0 or last?)
    # dataset.py line 208: activation[idx][0,1:] -> suggests index 0 is CLS/Register
    
    patches_features = act[0, 1:, :] # [N_patches, C]
    
    N_patches, C = patches_features.shape
    patch_grid_size = int(math.sqrt(N_patches))
    print(f"Grid size: {patch_grid_size}x{patch_grid_size} = {N_patches}")
    
    # Coordinates for each patch (center of the patch or simple grid index)
    # Grid coordinates: (0,0), (0,1), ..., (H-1, W-1)
    
    # Get object IDs for each patch
    # We need to resize instance_mask to patch grid size
    # instance_mask is [H_img, W_img]
    
    # Resize mask to (patch_grid_size, patch_grid_size) using geometric center
    # or mode.
    
    mask_tensor = torch.from_numpy(instance_mask).float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    # Resize nearest
    mask_small = torch.nn.functional.interpolate(mask_tensor, size=(patch_grid_size, patch_grid_size), mode='nearest')
    mask_small = mask_small.squeeze().cpu().numpy().astype(int) # [H_grid, W_grid]
    
    # Prepare data for plotting
    data_points = []
    
    # Collect all pairs
    # To avoid N^2 complexity if N is large (e.g. 1369 for 37x37), N^2 is ~2M, feasible.
    
    patches_features = patches_features.to(device)
    
    # Compute probe scores for all pairs
    # compute_batch_pairwise_similarity expects [B, N, C]. We have [N, C].
    # We can pass [1, N, C]
    
    print("Computing pairwise similarities...")
    pairwise_sim = compute_batch_pairwise_similarity(model, patches_features.unsqueeze(0), patches_features.unsqueeze(0))
    pairwise_sim = pairwise_sim.squeeze(0).detach().cpu().numpy() # [N, N]
    
    manhattan_points_blue = []
    manhattan_points_red = []
    
    euclidean_points_blue = []
    euclidean_points_red = []
    
    # Create grid coordinates
    xs, ys = np.meshgrid(np.arange(patch_grid_size), np.arange(patch_grid_size))
    xs = xs.flatten()
    ys = ys.flatten()
    mask_flat = mask_small.flatten()
    
    print("Collecting points...")
    # Iterate pairs
    for i in range(N_patches):
        for j in range(i + 1, N_patches): # Only upper triangle
            # Coordinates
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[j], ys[j]
            
            # Distances
            d_man = abs(x1 - x2) + abs(y1 - y2)
            d_euc = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            # Score
            score = pairwise_sim[i, j]
            
            # Class
            id1 = mask_flat[i]
            id2 = mask_flat[j]
            
            # Same object (Blue) if both are same instance AND not background (id != 0)
            if id1 == id2 and id1 != 0:
                is_same = True
            else:
                is_same = False
                
            if is_same:
                manhattan_points_blue.append((d_man, score))
                euclidean_points_blue.append((d_euc, score))
            else:
                manhattan_points_red.append((d_man, score))
                euclidean_points_red.append((d_euc, score))

    print(f"Blue pairs: {len(manhattan_points_blue)}")
    print(f"Red pairs: {len(manhattan_points_red)}")
    
    # Plotting Function
    def plot_graph(blue_pts, red_pts, title, xlabel, filename):
        plt.figure(figsize=(10, 8))
        
        # Red points (Different)
        if red_pts:
            rx, ry = zip(*red_pts)
            plt.scatter(rx, ry, c='red', alpha=0.1, s=1, label='Different Object')
            
            # Regression Red
            slope, intercept, r_value, p_value, std_err = linregress(rx, ry)
            line_x = np.array([min(rx), max(rx)])
            line_y = slope * line_x + intercept
            plt.plot(line_x, line_y, c='darkred', linewidth=2, label=f'Fit Different (R={r_value:.2f})')
        
        # Blue points (Same)
        if blue_pts:
            bx, by = zip(*blue_pts)
            plt.scatter(bx, by, c='blue', alpha=0.3, s=1, label='Same Object')
            
            # Regression Blue
            slope, intercept, r_value, p_value, std_err = linregress(bx, by)
            line_x = np.array([min(bx), max(bx)])
            line_y = slope * line_x + intercept
            plt.plot(line_x, line_y, c='darkblue', linewidth=2, label=f'Fit Same (R={r_value:.2f})')
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Probe Similarity Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved plot: {filename}")

    # Plot Manhattan
    plot_graph(manhattan_points_blue, manhattan_points_red, 
               "Manhattan Distance vs Similarity", "Manhattan Distance", "manhattan_plot.png")
               
    # Plot Euclidean
    plot_graph(euclidean_points_blue, euclidean_points_red, 
               "Euclidean Distance vs Similarity", "Euclidean Distance", "euclidean_plot.png")

if __name__ == "__main__":
    main()
