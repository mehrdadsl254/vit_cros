"""
Script to create test data in ADE20K format from two images with 4 car objects.
Image 1: A, B / C, D
Image 2: D, C / B, A
"""
import os
import json
import cv2
import numpy as np
from PIL import Image
import shutil
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available. Some advanced features may be limited.")

try:
    from skimage import segmentation, measure, morphology
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Note: scikit-image not available. Using OpenCV-based segmentation only.")

try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Note: Segment Anything Model (SAM) not available. Using OpenCV-based segmentation.")
    print("To install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git")

def load_sam_model(sam_checkpoint_path=None, model_type="vit_h", device="cuda"):
    """
    Load SAM model for segmentation.
    
    Args:
        sam_checkpoint_path: Path to SAM checkpoint. If None, will try to download or use default path.
        model_type: SAM model type - "vit_h" (best), "vit_l", or "vit_b" (fastest)
        device: Device to run on ("cuda" or "cpu")
    """
    if not SAM_AVAILABLE:
        return None, None
    
    if sam_checkpoint_path is None:
        # Try common paths
        possible_paths = [
            "checkpoints/sam_vit_h_4b8939.pth",
            "checkpoints/sam_vit_l_0b3195.pth",
            "checkpoints/sam_vit_b_01ec64.pth",
            os.path.expanduser("~/.cache/sam/sam_vit_h_4b8939.pth"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                sam_checkpoint_path = path
                break
        
        if sam_checkpoint_path is None:
            print("SAM checkpoint not found. Please download from:")
            print("https://github.com/facebookresearch/segment-anything#model-checkpoints")
            print("Or specify sam_checkpoint_path parameter.")
            return None, None
    
    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        if device == "cuda" and torch.cuda.is_available():
            sam.to(device=device)
            device_str = device
        else:
            sam.to(device="cpu")
            device_str = "cpu"
        sam.eval()
        
        # Create both predictor and automatic mask generator
        predictor = SamPredictor(sam)
        
        # Configure automatic mask generator for best quality
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,  # More points for better coverage
            pred_iou_thresh=0.88,  # High quality threshold
            stability_score_thresh=0.95,  # High stability requirement
            crop_n_layers=1,  # Crop layers for better detail
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Remove very small masks
        )
        
        return predictor, mask_generator
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        return None, None

def refine_mask_with_edges(img_rgb, mask, y0, y1, x0, x1):
    """
    Refine mask boundaries using edge detection and active contour techniques.
    This improves edge accuracy by using image gradients.
    """
    h, w = mask.shape
    refined_mask = mask.copy()
    
    # Extract region of interest
    roi_img = img_rgb[y0:y1, x0:x1] if len(img_rgb.shape) == 3 else img_rgb[y0:y1, x0:x1]
    roi_mask = mask[y0:y1, x0:x1]
    
    if roi_mask.sum() == 0:
        return refined_mask
    
    roi_h, roi_w = roi_mask.shape
    if roi_h == 0 or roi_w == 0:
        return refined_mask
    
    # Convert to grayscale for edge detection
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi_img
    
    # 1. Compute image gradients for edge detection
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient magnitude
    if grad_mag.max() > 0:
        grad_mag_norm = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
    else:
        grad_mag_norm = grad_mag.astype(np.uint8)
    
    # 2. Edge detection using Canny with adaptive thresholds
    median_val = np.median(gray)
    lower_thresh = int(max(0, 0.66 * median_val))
    upper_thresh = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(gray, lower_thresh, upper_thresh)
    
    # 3. Find contours of the mask
    contours, _ = cv2.findContours(roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return refined_mask
    
    # Use largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 4. Create distance map from edges for contour refinement
    edge_dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    
    # 5. Refine contour using active contour (snake) algorithm
    contour_points = largest_contour.reshape(-1, 2)
    
    if len(contour_points) >= 3:
        # Refine contour by moving points towards edges
        refined_contour = refine_contour_points(contour_points, edge_dist, gray, iterations=10)
        
        # Create refined mask from contour
        refined_roi_mask = np.zeros_like(roi_mask)
        cv2.fillPoly(refined_roi_mask, [refined_contour.reshape(-1, 1, 2).astype(np.int32)], 1)
    else:
        refined_roi_mask = roi_mask.copy()
    
    # 6. Use gradient-based boundary refinement
    # Create boundary region (dilated mask - original mask)
    boundary_mask = cv2.dilate(refined_roi_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) - refined_roi_mask
    boundary_coords = np.where(boundary_mask > 0)
    
    # Compute gradient threshold from object region
    if refined_roi_mask.sum() > 0:
        grad_thresh = np.percentile(grad_mag[refined_roi_mask > 0], 70)
    else:
        grad_thresh = np.percentile(grad_mag, 50)
    
    # Refine boundary pixels based on gradient strength
    final_roi_mask = refined_roi_mask.copy()
    
    for by, bx in zip(boundary_coords[0], boundary_coords[1]):
        if 0 <= by < grad_mag.shape[0] and 0 <= bx < grad_mag.shape[1]:
            grad_value = grad_mag[by, bx]
            
            # Strong gradient = likely edge, keep boundary
            # Weak gradient = might be noise, adjust based on neighbors
            if grad_value < grad_thresh * 0.3:
                # Check 3x3 neighborhood
                y_start, y_end = max(0, by-1), min(roi_h, by+2)
                x_start, x_end = max(0, bx-1), min(roi_w, bx+2)
                neighbors = final_roi_mask[y_start:y_end, x_start:x_end]
                
                # If majority of neighbors are inside, include this pixel
                if neighbors.sum() > neighbors.size * 0.6:
                    final_roi_mask[by, bx] = 1
                else:
                    final_roi_mask[by, bx] = 0
            # For strong gradients, keep the boundary as is
    
    # 7. Use watershed for final refinement (optional, can be slow)
    try:
        # Create markers
        markers = np.zeros_like(gray, dtype=np.int32)
        markers[final_roi_mask > 0] = 2  # Foreground
        markers[final_roi_mask == 0] = 1  # Background
        
        # Apply watershed only if we have a valid 3-channel image
        if len(gray.shape) == 2:
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            gray_3ch = gray
        
        cv2.watershed(gray_3ch, markers)
        watershed_mask = (markers == 2).astype(np.uint8)
        
        # Combine: prefer watershed but keep original if watershed is too different
        overlap = np.sum((final_roi_mask > 0) & (watershed_mask > 0)) / max(np.sum(final_roi_mask > 0), 1)
        if overlap > 0.7:  # If watershed result is similar, use it
            final_roi_mask = watershed_mask
    except Exception as e:
        # If watershed fails, continue with current mask
        pass
    
    # 8. Final cleanup: remove small holes and protrusions
    kernel = np.ones((3, 3), np.uint8)
    final_roi_mask = cv2.morphologyEx(final_roi_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    final_roi_mask = cv2.morphologyEx(final_roi_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Update the full mask
    refined_mask[y0:y1, x0:x1] = final_roi_mask
    
    return refined_mask

def refine_contour_points(contour_points, edge_dist, gray, iterations=10):
    """
    Refine contour points by moving them towards edges.
    Uses active contour (snake) algorithm with smoothness constraint.
    """
    if len(contour_points) < 3:
        return contour_points
    
    refined_points = contour_points.copy().astype(np.float32)
    n_points = len(refined_points)
    
    for iteration in range(iterations):
        new_points = refined_points.copy()
        
        for i in range(n_points):
            y, x = int(refined_points[i, 1]), int(refined_points[i, 0])
            
            # Clamp to image bounds
            y = max(0, min(gray.shape[0] - 1, y))
            x = max(0, min(gray.shape[1] - 1, x))
            
            # Search in neighborhood for best edge location
            search_radius = 5 if iteration < iterations // 2 else 3
            best_y, best_x = y, x
            best_score = edge_dist[y, x] if y < edge_dist.shape[0] and x < edge_dist.shape[1] else 0
            
            # Also consider smoothness (distance to neighbors)
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points
            prev_point = refined_points[prev_idx]
            next_point = refined_points[next_idx]
            center_point = (prev_point + next_point) / 2
            
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < edge_dist.shape[0] and 0 <= nx < edge_dist.shape[1]:
                        # Edge score
                        edge_score = edge_dist[ny, nx]
                        
                        # Smoothness score (prefer points near center of neighbors)
                        smoothness = 1.0 / (1.0 + np.sqrt((nx - center_point[0])**2 + (ny - center_point[1])**2))
                        
                        # Combined score
                        score = edge_score * 0.7 + smoothness * 0.3 * 10
                        
                        if score > best_score:
                            best_score = score
                            best_y, best_x = ny, nx
            
            # Smooth movement (gradually move towards best location)
            alpha = 0.3 + 0.2 * (iteration / iterations)  # Increase movement as we refine
            new_points[i, 0] = (1 - alpha) * refined_points[i, 0] + alpha * best_x
            new_points[i, 1] = (1 - alpha) * refined_points[i, 1] + alpha * best_y
        
        refined_points = new_points
    
    return refined_points.astype(np.int32)

def fill_holes_in_mask(mask):
    """
    Fill holes in a binary mask. Holes are white regions (255) surrounded by black (0).
    This fixes cases where blue parts (like car windows) create holes in the segmentation.
    """
    # Create a mask that is slightly larger than the original
    h, w = mask.shape
    mask_padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    mask_padded[1:h+1, 1:w+1] = mask
    
    # Create a mask for the background (holes + outer background)
    # Flood fill from the corners to mark all background
    mask_filled = mask_padded.copy()
    
    # Flood fill from all four corners to mark background
    cv2.floodFill(mask_filled, None, (0, 0), 255)
    cv2.floodFill(mask_filled, None, (w+1, 0), 255)
    cv2.floodFill(mask_filled, None, (0, h+1), 255)
    cv2.floodFill(mask_filled, None, (w+1, h+1), 255)
    
    # Invert: holes are now the regions that weren't filled (not background)
    mask_filled = 255 - mask_filled
    
    # Combine with original mask
    result = np.maximum(mask_padded, mask_filled)
    
    # Remove padding
    result = result[1:h+1, 1:w+1]
    
    # Additional morphological operations to smooth
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return result

def create_test_data(image1_path, image2_path, output_dir, base_name="test_car_rearrangement", 
                     sam_checkpoint_path=None, use_sam=True):
    """
    Create ADE20K format test data from two images.
    
    Args:
        image1_path: Path to first image (A, B / C, D arrangement)
        image2_path: Path to second image (D, C / B, A arrangement)
        output_dir: Directory to save test data
        base_name: Base name for the test files
        sam_checkpoint_path: Path to SAM checkpoint (optional)
        use_sam: Whether to use SAM for segmentation (best quality)
    """
    # Create directory structure
    test_folder = os.path.join(output_dir, "ADE", "training", "test_cars", f"{base_name}")
    os.makedirs(test_folder, exist_ok=True)
    instance_folder = os.path.join(test_folder, f"{base_name}")
    os.makedirs(instance_folder, exist_ok=True)
    
    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        raise ValueError(f"Could not load images from {image1_path} or {image2_path}")
    
    h, w = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    # Load SAM model if available and requested
    sam_predictor = None
    sam_mask_generator = None
    if use_sam and SAM_AVAILABLE:
        print("Loading SAM (Segment Anything Model) for best segmentation quality...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_predictor, sam_mask_generator = load_sam_model(sam_checkpoint_path, model_type="vit_h", device=device)
        if sam_predictor is not None:
            print("SAM model loaded successfully!")
        else:
            print("Falling back to OpenCV-based segmentation.")
    
    def segment_objects_by_color(img, arrangement):
        """
        Simple and effective color-based segmentation.
        Background is blue, objects are not blue - very straightforward!
        """
        h, w = img.shape[:2]
        instance_mask = np.zeros((h, w), dtype=np.uint16)
        class_mask = np.zeros((h, w), dtype=np.uint8)
        car_class_id = 14
        
        # Convert to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Convert to HSV for better color segmentation
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Define blue color range in HSV
        # Blue in HSV: Hue around 100-130 (out of 180)
        lower_blue = np.array([100, 50, 50])   # Lower bound for blue
        upper_blue = np.array([130, 255, 255]) # Upper bound for blue
        
        # Create mask for blue background
        blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
        
        # Invert to get non-blue (objects)
        objects_mask = 255 - blue_mask
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find connected components (each object)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            objects_mask, connectivity=8
        )
        
        # Define quadrants
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            (0, mid_h, 0, mid_w),           # top-left
            (0, mid_h, mid_w, w),           # top-right
            (mid_h, h, 0, mid_w),           # bottom-left
            (mid_h, h, mid_w, w)            # bottom-right
        ]
        
        # Assign objects to quadrants based on centroid and fill holes
        for label_id in range(1, num_labels):  # Skip background (0)
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < 100:  # Skip very small components
                continue
            
            centroid_x, centroid_y = centroids[label_id]
            
            # Find which quadrant this object belongs to
            for idx, (obj_id, (y0, y1, x0, x1)) in enumerate(zip(arrangement, quadrants)):
                if x0 <= centroid_x <= x1 and y0 <= centroid_y <= y1:
                    # This object belongs to this quadrant
                    obj_mask = (labels == label_id).astype(np.uint8)
                    
                    # Fill holes in the object mask
                    obj_mask = fill_holes_in_mask(obj_mask)
                    
                    instance_mask[obj_mask > 0] = obj_id
                    class_mask[obj_mask > 0] = car_class_id
                    break
        
        return instance_mask, class_mask
    
    def segment_objects_with_sam_advanced(img, arrangement, sam_predictor, sam_mask_generator):
        """
        Segment objects using SAM with advanced techniques for best quality:
        - Automatic mask generation for all objects
        - Point + box prompts for each quadrant
        - Advanced post-processing and refinement
        """
        h, w = img.shape[:2]
        instance_mask = np.zeros((h, w), dtype=np.uint16)
        class_mask = np.zeros((h, w), dtype=np.uint8)
        car_class_id = 14
        
        # Convert to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Define quadrants with large overlap to capture full objects
        mid_h, mid_w = h // 2, w // 2
        # Use large expansion (40-50%) to ensure we capture full objects including sides
        expand_h = int(h * 0.25)  # 25% expansion vertically
        expand_w = int(w * 0.25)  # 25% expansion horizontally
        
        quadrants = [
            (max(0, 0 - expand_h), min(h, mid_h + expand_h), 
             max(0, 0 - expand_w), min(w, mid_w + expand_w)),           # top-left (expanded)
            (max(0, 0 - expand_h), min(h, mid_h + expand_h), 
             max(0, mid_w - expand_w), min(w, w + expand_w)),           # top-right (expanded)
            (max(0, mid_h - expand_h), min(h, h + expand_h), 
             max(0, 0 - expand_w), min(w, mid_w + expand_w)),           # bottom-left (expanded)
            (max(0, mid_h - expand_h), min(h, h + expand_h), 
             max(0, mid_w - expand_w), min(w, w + expand_w))            # bottom-right (expanded)
        ]
        
        # Original quadrant centers for reference
        original_quadrants = [
            (0, mid_h, 0, mid_w),           # top-left
            (0, mid_h, mid_w, w),           # top-right
            (mid_h, h, 0, mid_w),           # bottom-left
            (mid_h, h, mid_w, w)            # bottom-right
        ]
        
        # First, use automatic mask generator to get all masks
        all_masks = []
        if sam_mask_generator is not None:
            print("  Using SAM automatic mask generator...")
            auto_masks = sam_mask_generator.generate(img_rgb)
            all_masks = auto_masks
        
        # Set image for predictor
        sam_predictor.set_image(img_rgb)
        
        for idx, (obj_id, (y0, y1, x0, x1)) in enumerate(zip(arrangement, quadrants)):
            # Get original quadrant for reference
            orig_y0, orig_y1, orig_x0, orig_x1 = original_quadrants[idx]
            orig_center_y = (orig_y0 + orig_y1) // 2
            orig_center_x = (orig_x0 + orig_x1) // 2
            
            qh = y1 - y0
            qw = x1 - x0
            center_y = (y0 + y1) // 2
            center_x = (x0 + x1) // 2
            
            best_mask = None
            best_score = -1
            
            # Strategy 1: Use automatic masks that overlap with quadrant (very permissive)
            if len(all_masks) > 0:
                for mask_info in all_masks:
                    mask = mask_info['segmentation']
                    bbox = mask_info['bbox']  # [x, y, w, h]
                    mask_x, mask_y, mask_w, mask_h = bbox
                    mask_center_x = mask_x + mask_w // 2
                    mask_center_y = mask_y + mask_h // 2
                    
                    # Check if mask center is in the expanded search region
                    if y0 <= mask_center_y <= y1 and x0 <= mask_center_x <= x1:
                        # Check overlap with original quadrant (very permissive - only 10% required)
                        orig_quadrant_mask = np.zeros((h, w), dtype=bool)
                        orig_quadrant_mask[orig_y0:orig_y1, orig_x0:orig_x1] = True
                        overlap_with_orig = np.sum(mask & orig_quadrant_mask) / max(np.sum(mask), 1)
                        
                        # Check overlap with expanded search region
                        search_mask = np.zeros((h, w), dtype=bool)
                        search_mask[y0:y1, x0:x1] = True
                        overlap_with_search = np.sum(mask & search_mask) / max(np.sum(mask), 1)
                        
                        # Very permissive: accept if any overlap with original quadrant OR good overlap with search region
                        if overlap_with_orig > 0.1 or overlap_with_search > 0.2:
                            # Score based on both overlaps, but prioritize original quadrant slightly
                            score = (mask_info.get('predicted_iou', 0.5) * 
                                   mask_info.get('stability_score', 0.5) * 
                                   (0.5 * overlap_with_orig + 0.5 * overlap_with_search))
                            if score > best_score:
                                best_score = score
                                best_mask = mask
            
            # Strategy 2: Use point prompts with multiple points across the full search region
            if best_mask is None or best_score < 0.7:
                # Try many points across the entire search region to capture full object
                # Sample points in a grid pattern across the search region
                num_points_x = 5
                num_points_y = 5
                step_x = qw // (num_points_x - 1) if num_points_x > 1 else 1
                step_y = qh // (num_points_y - 1) if num_points_y > 1 else 1
                
                points_to_try = []
                # Grid of points across search region
                for i in range(num_points_y):
                    for j in range(num_points_x):
                        px = x0 + j * step_x
                        py = y0 + i * step_y
                        if 0 <= px < w and 0 <= py < h:
                            points_to_try.append([px, py])
                
                # Also try original quadrant center and corners
                points_to_try.extend([
                    [orig_center_x, orig_center_y],  # Original quadrant center
                    [orig_x0 + orig_x1//4, orig_center_y],  # Left of original
                    [orig_x0 + 3*orig_x1//4, orig_center_y],  # Right of original
                    [orig_center_x, orig_y0 + orig_y1//4],  # Top of original
                    [orig_center_x, orig_y0 + 3*orig_y1//4],  # Bottom of original
                ])
                
                for point in points_to_try:
                    input_point = np.array([point])
                    input_label = np.array([1])
                    
                    # Use the full search region as box (very permissive)
                    search_box = np.array([x0, y0, x1, y1])
                    
                    try:
                        masks, scores, logits = sam_predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            box=search_box,
                            multimask_output=True,
                        )
                        
                        mask_idx = np.argmax(scores)
                        score = scores[mask_idx]
                        
                        # Check overlap with original quadrant (very permissive - only 10% required)
                        mask_area_in_orig = np.sum(masks[mask_idx][orig_y0:orig_y1, orig_x0:orig_x1])
                        mask_total_area = np.sum(masks[mask_idx])
                        orig_ratio = mask_area_in_orig / max(mask_total_area, 1)
                        
                        # Check overlap with search region
                        mask_area_in_search = np.sum(masks[mask_idx][y0:y1, x0:x1])
                        search_ratio = mask_area_in_search / max(mask_total_area, 1)
                        
                        # Very permissive: accept if any overlap with original quadrant
                        if orig_ratio > 0.1 or search_ratio > 0.2:
                            # Score based on both, but don't penalize too much for extension
                            adjusted_score = score * (0.4 * orig_ratio + 0.6 * search_ratio + 0.2)
                            if adjusted_score > best_score:
                                best_score = adjusted_score
                                best_mask = masks[mask_idx]
                    except Exception as e:
                        continue
            
            # Strategy 3: Use box-only prompt with full search region
            if best_mask is None or best_score < 0.6:
                try:
                    # Use the full search region as box (already expanded)
                    search_box = np.array([x0, y0, x1, y1])
                    masks, scores, logits = sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=search_box,
                        multimask_output=True,
                    )
                    mask_idx = np.argmax(scores)
                    score = scores[mask_idx]
                    
                    # Very permissive: check if mask overlaps with original quadrant (only 10% required)
                    mask_area_in_orig = np.sum(masks[mask_idx][orig_y0:orig_y1, orig_x0:orig_x1])
                    mask_area_in_search = np.sum(masks[mask_idx][y0:y1, x0:x1])
                    
                    if mask_area_in_orig > (orig_y1 - orig_y0) * (orig_x1 - orig_x0) * 0.1 or mask_area_in_search > (qh * qw * 0.15):
                        if score > best_score:
                            best_score = score
                            best_mask = masks[mask_idx]
                except Exception as e:
                    pass
            
            # Fallback: use search region if no good mask found
            if best_mask is None:
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y0:y1, x0:x1] = 1
            else:
                mask = best_mask.astype(np.uint8)
                # Use the FULL mask - don't crop at all to capture complete objects
                # The mask can extend in all directions (left, right, top, bottom)
                # Only ensure it has some overlap with original quadrant
                
                # Check overlap with original quadrant
                overlap_with_orig = np.sum(mask[orig_y0:orig_y1, orig_x0:orig_x1])
                
                # If mask has at least 10% overlap with original quadrant, use it fully
                # This allows objects to extend fully in all directions
                if overlap_with_orig < (orig_y1 - orig_y0) * (orig_x1 - orig_x0) * 0.1:
                    # If overlap is too small, try to find the part that overlaps
                    # But still allow extension
                    mask_cropped = np.zeros((h, w), dtype=np.uint8)
                    # Use full mask but ensure it touches original quadrant
                    mask_cropped = mask.copy()
                    
                    # If no overlap, fall back to search region
                    if overlap_with_orig == 0:
                        mask_cropped = np.zeros((h, w), dtype=np.uint8)
                        mask_cropped[y0:y1, x0:x1] = 1
                    
                    mask = mask_cropped
                else:
                    # Good overlap - use full mask as-is (no cropping)
                    mask = mask.copy()
            
            # Advanced post-processing with edge-aware refinement
            # 1. Remove small components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                # Keep largest component
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_label = 1 + np.argmax(areas)
                mask = (labels == largest_label).astype(np.uint8)
            
            # 2. Edge-aware boundary refinement
            mask = refine_mask_with_edges(img_rgb, mask, y0, y1, x0, x1)
            
            # 3. Fill holes using morphological operations (but preserve edges)
            kernel_small = np.ones((3, 3), np.uint8)
            kernel_medium = np.ones((5, 5), np.uint8)
            
            # Close small gaps (but be careful not to blur edges)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
            
            # 4. Remove small protrusions
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
            
            # 5. Final cleanup - remove very small remaining components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                min_area = (qh * qw) * 0.05  # At least 5% of quadrant
                for label_id in range(1, num_labels):
                    if stats[label_id, cv2.CC_STAT_AREA] < min_area:
                        mask[labels == label_id] = 0
            
            # Apply to instance and class masks
            instance_mask[mask > 0] = obj_id
            class_mask[mask > 0] = car_class_id
        
        return instance_mask, class_mask
    
    def segment_objects_professional(img, arrangement):
        """
        Professionally segment objects from image using advanced techniques.
        Returns instance mask and class mask with proper object boundaries.
        """
        h, w = img.shape[:2]
        instance_mask = np.zeros((h, w), dtype=np.uint16)
        class_mask = np.zeros((h, w), dtype=np.uint8)
        car_class_id = 14
        
        # Convert to RGB if BGR
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Method 1: Use GrabCut for each quadrant to separate foreground from background
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            (0, mid_h, 0, mid_w),           # top-left
            (0, mid_h, mid_w, w),           # top-right
            (mid_h, h, 0, mid_w),           # bottom-left
            (mid_h, h, mid_w, w)            # bottom-right
        ]
        
        for idx, (obj_id, (y0, y1, x0, x1)) in enumerate(zip(arrangement, quadrants)):
            # Extract quadrant
            quadrant = img_rgb[y0:y1, x0:x1].copy()
            qh, qw = quadrant.shape[:2]
            
            if qh == 0 or qw == 0:
                continue
            
            # Create mask for GrabCut
            mask = np.zeros((qh, qw), np.uint8)
            # Initialize with probable foreground in center region
            center_y, center_x = qh // 2, qw // 2
            center_size = min(qh, qw) // 3
            cv2.rectangle(mask, 
                         (center_x - center_size, center_y - center_size),
                         (center_x + center_size, center_y + center_size),
                         cv2.GC_PR_FGD, -1)
            # Border as background
            border = 10
            cv2.rectangle(mask, (0, 0), (qw, qh), cv2.GC_BGD, border)
            
            # Prepare for GrabCut
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # Run GrabCut
            try:
                cv2.grabCut(quadrant, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                
                # Create binary mask (foreground = 1, background = 0)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                
                # Refine using morphological operations
                kernel = np.ones((5, 5), np.uint8)
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
                
                # Remove small connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask2, connectivity=8)
                if num_labels > 1:
                    # Find largest component (should be the car)
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    mask2 = (labels == largest_label).astype('uint8')
                
                # Apply mask to instance and class masks
                instance_mask[y0:y1, x0:x1][mask2 > 0] = obj_id
                class_mask[y0:y1, x0:x1][mask2 > 0] = car_class_id
                
            except Exception as e:
                print(f"Warning: GrabCut failed for object {obj_id}, using quadrant fallback: {e}")
                # Fallback: use entire quadrant
                instance_mask[y0:y1, x0:x1] = obj_id
                class_mask[y0:y1, x0:x1] = car_class_id
        
        # Post-process: Use watershed to better separate objects if they're touching
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if len(img_rgb.shape) == 3 else img_rgb
        
        # Create distance transform from instance mask boundaries
        for obj_id in arrangement:
            obj_mask = (instance_mask == obj_id).astype(np.uint8)
            if obj_mask.sum() == 0:
                continue
            
            # Find contours and refine
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                # Use largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Create refined mask from contour
                refined_mask = np.zeros_like(obj_mask)
                cv2.fillPoly(refined_mask, [largest_contour], 1)
                
                # Smooth the boundary
                refined_mask = cv2.GaussianBlur(refined_mask.astype(np.float32), (5, 5), 0)
                refined_mask = (refined_mask > 0.5).astype(np.uint8)
                
                # Update instance mask
                instance_mask[refined_mask > 0] = obj_id
                class_mask[refined_mask > 0] = car_class_id
        
        return instance_mask, class_mask

    
    def segment_objects_advanced(img, arrangement):
        """
        Alternative advanced segmentation using multiple techniques.
        """
        h, w = img.shape[:2]
        instance_mask = np.zeros((h, w), dtype=np.uint16)
        class_mask = np.zeros((h, w), dtype=np.uint8)
        car_class_id = 14
        
        # Convert to appropriate color space
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        else:
            img_rgb = img
            img_lab = img
        
        # Divide into quadrants
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            (0, mid_h, 0, mid_w),
            (0, mid_h, mid_w, w),
            (mid_h, h, 0, mid_w),
            (mid_h, h, mid_w, w)
        ]
        
        for idx, (obj_id, (y0, y1, x0, x1)) in enumerate(zip(arrangement, quadrants)):
            quadrant_rgb = img_rgb[y0:y1, x0:x1].copy()
            quadrant_lab = img_lab[y0:y1, x0:x1].copy() if len(img_lab.shape) == 3 else quadrant_rgb
            qh, qw = quadrant_rgb.shape[:2]
            
            if qh == 0 or qw == 0:
                continue
            
            # Method: Use k-means clustering to separate object from background
            # Reshape for k-means
            data = quadrant_rgb.reshape((-1, 3))
            data = np.float32(data)
            
            # K-means with 2 clusters (object and background)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            k = 2
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Reshape labels
            labels = labels.reshape((qh, qw))
            
            # Determine which cluster is the object (usually the one not at edges)
            edge_mask = np.zeros((qh, qw), dtype=bool)
            edge_mask[0:5, :] = True
            edge_mask[-5:, :] = True
            edge_mask[:, 0:5] = True
            edge_mask[:, -5:] = True
            
            cluster0_in_edge = np.sum(labels[edge_mask] == 0)
            cluster1_in_edge = np.sum(labels[edge_mask] == 1)
            
            # Object cluster is the one with fewer pixels at edges
            if cluster0_in_edge < cluster1_in_edge:
                obj_cluster = 0
            else:
                obj_cluster = 1
            
            obj_mask = (labels == obj_cluster).astype(np.uint8)
            
            # Refine mask
            kernel = np.ones((3, 3), np.uint8)
            obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Remove small components
            num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                obj_mask = (labels_cc == largest_label).astype(np.uint8)
            
            # Apply to full image masks
            instance_mask[y0:y1, x0:x1][obj_mask > 0] = obj_id
            class_mask[y0:y1, x0:x1][obj_mask > 0] = car_class_id
        
        return instance_mask, class_mask
    
    # Image 1: A=1, B=2, C=3, D=4
    print("\n" + "="*60)
    print("Segmenting objects in Image 1...")
    print("="*60)
    print("Using simple color-based segmentation (blue background detection)...")
    instance_mask1, class_mask1 = segment_objects_by_color(img1, [1, 2, 3, 4])
    
    # Image 2: D=4, C=3, B=2, A=1
    print("\n" + "="*60)
    print("Segmenting objects in Image 2...")
    print("="*60)
    print("Using simple color-based segmentation (blue background detection)...")
    instance_mask2, class_mask2 = segment_objects_by_color(img2,  [4, 3, 2, 1])
    
    # Print segmentation statistics
    print(f"\nSegmentation Statistics:")
    print(f"Image 1 - Pixels segmented: {np.sum(instance_mask1 > 0)} / {h * w} ({100 * np.sum(instance_mask1 > 0) / (h * w):.1f}%)")
    print(f"Image 2 - Pixels segmented: {np.sum(instance_mask2 > 0)} / {h * w} ({100 * np.sum(instance_mask2 > 0) / (h * w):.1f}%)")
    print(f"Image 1 - Objects found: {len(np.unique(instance_mask1[instance_mask1 > 0]))}")
    print(f"Image 2 - Objects found: {len(np.unique(instance_mask2[instance_mask2 > 0]))}")
    # print(f"------------------------------------h --w -------------{h},{w}----2-------{h2},{w2}-----")
    # Save images
    img1_name = f"{base_name}_img1.jpg"
    img2_name = f"{base_name}_img2.jpg"
    cv2.imwrite(os.path.join(test_folder, img1_name), img1)
    cv2.imwrite(os.path.join(test_folder, img2_name), img2)
    
    # Save segmentation masks
    seg1_name = f"{base_name}_img1_seg.png"
    seg2_name = f"{base_name}_img2_seg.png"
    cv2.imwrite(os.path.join(test_folder, seg1_name), class_mask1)
    cv2.imwrite(os.path.join(test_folder, seg2_name), class_mask2)
    
    # Save instance masks (as individual files and combined)
    def save_instance_masks(instance_mask, base_path, base_name_img):
        """Save individual instance mask files"""
        unique_ids = np.unique(instance_mask)
        unique_ids = unique_ids[unique_ids > 0]  # Exclude background
        
        for obj_id in unique_ids:
            mask = (instance_mask == obj_id).astype(np.uint8) * 255
            instance_file = os.path.join(base_path, f"instance_{obj_id-1:03d}_{base_name_img}.png")
            cv2.imwrite(instance_file, mask)
    save_instance_masks(instance_mask1, instance_folder, f"{base_name}_img1")
    save_instance_masks(instance_mask2, instance_folder, f"{base_name}_img2")
    
    # Create JSON annotations for Image 1
    def create_json_annotation(img_name, seg_name, instance_mask, class_mask, h, w, base_name_img):
        """Create ADE20K JSON annotation"""
        objects = []
        unique_ids = np.unique(instance_mask)
        unique_ids = unique_ids[unique_ids > 0]
        
        object_names = {1: "car", 2: "car", 3: "car", 4: "car"}
        
        for obj_id in unique_ids:
            mask = (instance_mask == obj_id).astype(np.uint8)
            y_coords, x_coords = np.where(mask > 0)
            
            if len(x_coords) == 0:
                continue
            
            # Get actual contour for more accurate polygon
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Use largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Simplify contour (reduce points)
                epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Extract polygon points
                if len(approx_contour) > 0:
                    polygon_points = approx_contour.reshape(-1, 2)
                    polygon_x = polygon_points[:, 0].tolist()
                    polygon_y = polygon_points[:, 1].tolist()
                    # Close the polygon
                    if polygon_x[0] != polygon_x[-1] or polygon_y[0] != polygon_y[-1]:
                        polygon_x.append(polygon_x[0])
                        polygon_y.append(polygon_y[0])
                else:
                    # Fallback to bounding box
                    x_min, x_max = int(x_coords.min()), int(x_coords.max())
                    y_min, y_max = int(y_coords.min()), int(y_coords.max())
                    polygon_x = [x_min, x_max, x_max, x_min, x_min]
                    polygon_y = [y_min, y_min, y_max, y_max, y_min]
            else:
                # Fallback to bounding box
                x_min, x_max = int(x_coords.min()), int(x_coords.max())
                y_min, y_max = int(y_coords.min()), int(y_coords.max())
                polygon_x = [x_min, x_max, x_max, x_min, x_min]
                polygon_y = [y_min, y_min, y_max, y_max, y_min]
            
            obj_name = object_names.get(obj_id, "car")
            instance_file = f"{base_name_img}/instance_{obj_id-1:03d}_{base_name_img}.png"
            
            obj_dict = {
                "id": int(obj_id - 1),  # ADE20K uses 0-indexed
                "name": obj_name,
                "name_ndx": 14,  # Car class index
                "hypernym": ["car", "vehicle", "automobile"],
                "raw_name": obj_name,
                "attributes": [],
                "depth_ordering_rank": int(obj_id),
                "occluded": [],
                "crop": "0",
                "parts": {
                    "hasparts": [],
                    "ispartof": [],
                    "part_level": 0
                },
                "instance_mask": instance_file,
                "polygon": {
                    "x": polygon_x,
                    "y": polygon_y,
                    "click_date": []
                },
                "saved_date": ""
            }
            objects.append(obj_dict)
        
        annotation = {
            "annotation": {
                "filename": img_name,
                "folder": f"ADE20K_2021_17_01/images/ADE/training/test_cars/{base_name_img}",
                "imsize": [h, w, 3],
                "source": {
                    "folder": "test_data",
                    "filename": img_name,
                    "origin": "Generated test data"
                },
                "scene": ["test", "cars"],
                "object": objects
            }
        }
        
        return annotation
    
    json1 = create_json_annotation(img1_name, seg1_name, instance_mask1, class_mask1, h, w, f"{base_name}_img1")
    json2 = create_json_annotation(img2_name, seg2_name, instance_mask2, class_mask2, h, w, f"{base_name}_img2")
    
    # Save JSON files
    json1_name = f"{base_name}_img1.json"
    json2_name = f"{base_name}_img2.json"
    
    with open(os.path.join(test_folder, json1_name), 'w') as f:
        json.dump(json1, f, indent=2)
    
    with open(os.path.join(test_folder, json2_name), 'w') as f:
        json.dump(json2, f, indent=2)
    
    # Optional: Save visualization of segmentation
    try:
        import matplotlib.pyplot as plt
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Image 1
        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Image 1 - Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(instance_mask1, cmap='tab10')
        axes[0, 1].set_title('Image 1 - Instance Mask')
        axes[0, 1].axis('off')
        
        overlay1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).copy()
        for obj_id in [1, 2, 3, 4]:
            mask = (instance_mask1 == obj_id)
            overlay1[mask, 0] = np.clip(overlay1[mask, 0] * 0.7 + 255 * 0.3, 0, 255)
        axes[0, 2].imshow(overlay1)
        axes[0, 2].set_title('Image 1 - Overlay')
        axes[0, 2].axis('off')
        
        # Image 2
        axes[1, 0].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Image 2 - Original')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(instance_mask2, cmap='tab10')
        axes[1, 1].set_title('Image 2 - Instance Mask')
        axes[1, 1].axis('off')
        
        overlay2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).copy()
        for obj_id in [1, 2, 3, 4]:
            mask = (instance_mask2 == obj_id)
            overlay2[mask, 0] = np.clip(overlay2[mask, 0] * 0.7 + 255 * 0.3, 0, 255)
        axes[1, 2].imshow(overlay2)
        axes[1, 2].set_title('Image 2 - Overlay')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        vis_path = os.path.join(test_folder, f"{base_name}_segmentation_visualization.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Segmentation visualization saved to: {vis_path}")
    except ImportError:
        print("matplotlib not available, skipping visualization")
    
    print(f"\nCreated test data in {test_folder}")
    print(f"Image 1: {img1_name} (A=1, B=2, C=3, D=4)")
    print(f"Image 2: {img2_name} (D=4, C=3, B=2, A=1)")
    
    return test_folder, instance_mask1, instance_mask2, class_mask1, class_mask2



#################6 image part #####################################################################################################
def segment_single_object(img, use_sam=True, sam_predictor=None, sam_mask_generator=None):
    """
    Segment a single main object from an image.
    
    Args:
        img: Input image (BGR or RGB)
        use_sam: Whether to use SAM if available
        sam_predictor: Pre-loaded SAM predictor
        sam_mask_generator: Pre-loaded SAM mask generator
        
    Returns:
        instance_mask: Mask with 1 where the object is, 0 elsewhere
        class_mask: Mask with class ID (14 for car) where the object is, 0 elsewhere
    """
    h, w = img.shape[:2]
    instance_mask = np.zeros((h, w), dtype=np.uint16)
    class_mask = np.zeros((h, w), dtype=np.uint8)
    car_class_id = 14
    
    # Convert to RGB if BGR
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
        
    object_mask = None
    
    # Try SAM first
    if use_sam and sam_predictor is not None:
        try:
            sam_predictor.set_image(img_rgb)
            
            # Prompt with center point
            input_point = np.array([[w // 2, h // 2]])
            input_label = np.array([1])
            
            # Also prompt with a loose box in the center 50%
            box = np.array([w//4, h//4, 3*w//4, 3*h//4])
            
            masks, scores, logits = sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=box,
                multimask_output=True,
            )
            
            # Pick best mask
            best_idx = np.argmax(scores)
            object_mask = masks[best_idx].astype(np.uint8)
            
        except Exception as e:
            print(f"SAM segmentation failed: {e}")
            object_mask = None
            
    # Fallback to color/contour if SAM not used or failed
    if object_mask is None:
        # Simple color based + keep largest component
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        # Assuming blue background
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
        fg_mask = 255 - blue_mask
        
        # Keep largest component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            fg_mask, connectivity=8
        )
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            object_mask = (labels == largest_label).astype(np.uint8)
        else:
            object_mask = fg_mask > 0
            
    # Final cleanup
    if object_mask is not None:
        object_mask = object_mask.astype(np.uint8)
        
        # Fill holes
        object_mask = fill_holes_in_mask(object_mask) 
        
        # Assign to masks
        # For single object, we assign ID 1
        instance_mask[object_mask > 0] = 1
        class_mask[object_mask > 0] = car_class_id
        
    return instance_mask, class_mask

#################6 image part end#####################################################################################################


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create ADE20K format test data with professional segmentation")
    parser.add_argument("--image1", type=str, 
                       default="/Users/mehrdadsalehi/Anki/doesp/vit-object-binding/vit-object-binding/vit-object-binding/image1.png",
                       help="Path to first image (A, B / C, D)")
    parser.add_argument("--image2", type=str,
                       default="/Users/mehrdadsalehi/Anki/doesp/vit-object-binding/vit-object-binding/vit-object-binding/image2.png",
                       help="Path to second image (D, C / B, A)")
    parser.add_argument("--output_dir", type=str, default="../../", help="Output directory")
    parser.add_argument("--sam_checkpoint", type=str, default=None, 
                       help="Path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth). If not provided, will try to find automatically.")
    parser.add_argument("--no_sam", action="store_true", 
                       help="Disable SAM and use OpenCV-based segmentation")
    
    args = parser.parse_args()
    
    create_test_data(
        args.image1, 
        args.image2, 
        args.output_dir,
        sam_checkpoint_path=args.sam_checkpoint,
        use_sam=not args.no_sam
    )

