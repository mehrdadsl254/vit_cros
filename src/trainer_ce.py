import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import wandb # type: ignore

from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.dataset import ADE20KSegmentation, FeatureExtractor, collate_fn
from utils.models import get_model
from utils.score import compute_batch_pairwise_similarity
from utils.utils import set_random_seed



class TrainerCE:
    def __init__(self, cfg, output_dir):
        
        set_random_seed(cfg.seed)
        self.model = get_model(cfg)
        
        dataset = ADE20KSegmentation(root=cfg.dataset.data_dir, image_set="train")
        
        train_size = int(cfg.dataset.train_ratio * len(dataset))
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, len(dataset)))
        

        remove_indices = {7689, 8904, 8905, 20433, 20543, 21029, 22979, 23011, 23048, 24103, 24794}
        # Remove these indices from both train_indices and val_indices
        train_indices = [idx for idx in train_indices if idx not in remove_indices]
        val_indices = [idx for idx in val_indices if idx not in remove_indices]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = ADE20KSegmentation(root=cfg.dataset.data_dir, image_set="val")
        
        self.train_dataloader = DataLoader(train_dataset, batch_size = cfg.trainer.batch_size, shuffle = True, num_workers = cfg.trainer.num_workers, collate_fn = collate_fn)
        self.val_dataloader = DataLoader(val_dataset, batch_size = cfg.trainer.batch_size, shuffle = False, num_workers = cfg.trainer.num_workers, collate_fn = collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size = cfg.trainer.batch_size, shuffle = False, num_workers = cfg.trainer.num_workers, collate_fn = collate_fn)
        
        
        print('train_dataset size:' + str(len(self.train_dataloader)))
        print('val_dataset size:' + str(len(self.val_dataloader)))
        print('test_dataset size:' + str(len(self.test_dataloader)))
        
        self.feature_extractor = FeatureExtractor(cfg, test_dataset.class_mapping)

        self.device = cfg.trainer.device
        self.model = self.model.to(self.device)
        self.model = self.model.to(self.device)
        # print(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(cfg.trainer.learning_rate))
        self.scheduler = StepLR(self.optimizer, step_size=cfg.trainer.scheduler.step_size, gamma=cfg.trainer.scheduler.gamma) 
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.output_dir = output_dir

        # Load checkpoint if exists
        if cfg.mode == 'train' or cfg.mode == 'train_ce':
            CHECKPOINT_PATH = os.path.join(output_dir, 'checkpoint.pth')
        elif cfg.mode == 'eval_all':
            CHECKPOINT_PATH = os.path.join(output_dir, 'best_checkpoint.pth')
            
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']+1
            self.best_val = checkpoint['best_val']
            self.best_test = checkpoint['best_test']
            print("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, self.start_epoch))
                
        else:
            self.start_epoch = 0
            self.best_val = 1e5
            self.best_test = 1e5
        
    
    def train(self, cfg):
        
        for epoch in range(self.start_epoch, cfg.trainer.max_epoch):
            stat_dict = {}
            
            stat_dict['train_loss'] = 0.0
            self.model.train()
            correct = 0
            total_num = 0
            total_updates = 0
            for images, masks, bboxes in tqdm(self.train_dataloader):
                features, labels, _ = self.feature_extractor.forward(images, masks, bboxes)
                labels = labels.to(self.device)
                features = features.to(self.device)
                self.optimizer.zero_grad()
                pairwise_similarity = compute_batch_pairwise_similarity(self.model, features, features)
                
                # pairwise_similarity > 0.5 -> same object -> label = 1, consider all N*N pairs
                predicted = (pairwise_similarity > 0.0).float()
                labels_pairwise = labels.unsqueeze(1) == labels.unsqueeze(2)#[B,N,N]
                #import pdb; pdb.set_trace()
                loss = self.criterion(pairwise_similarity.reshape(-1), labels_pairwise.reshape(-1).float())
                loss.backward()
                

                #print(f"Gradient Norm: {grad_norm.item()}")
                self.optimizer.step()
                
                correct += torch.sum(torch.triu(predicted == labels_pairwise)).item()
                stat_dict['train_loss'] += loss.item() * predicted.numel() # B * N
                total_num += torch.triu(torch.ones_like(predicted)).sum().item()
                total_updates += predicted.numel()

            stat_dict['train_acc'] = correct / total_num
            for key in ['train_loss']:
                stat_dict[key] = stat_dict[key] /total_updates
            print(f'[Epoch {epoch+1}] loss: {stat_dict["train_loss"]:.4f}, acc: {stat_dict["train_acc"]*100:.4f}%')
            # Save current checkpoint to resume training
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val': self.best_val,
                'best_test': self.best_test
            }
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint.pth")
            torch.save(checkpoint, checkpoint_path)
            
            self.scheduler.step()
            
            val_performance = self.eval(cfg, epoch, split='val')
            test_performance = self.eval(cfg, epoch, split='test')
            stat_dict.update(val_performance)
            stat_dict.update(test_performance)

            if self.best_val >= stat_dict['train_loss']:
                self.best_val = stat_dict['val_loss']
                self.best_test = stat_dict['test_loss']
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val': self.best_val,
                    'best_test': self.best_test
                }
                checkpoint_path = os.path.join(self.output_dir, f"best_checkpoint.pth")
                torch.save(checkpoint, checkpoint_path)
            stat_dict['epoch'] = epoch + 1
            wandb.log(stat_dict)
        
    
    def eval(self, cfg, epoch, split = 'val'):
        stat_dict = {}
        stat_dict[split + '_loss'] = 0.0

        self.model.eval()
        
        if split == 'val':
            dataloader = self.val_dataloader
        elif split == 'test':
            dataloader = self.test_dataloader
        elif split == 'train':
            dataloader = self.train_dataloader
        with torch.no_grad():
            correct = 0
            total_num = 0
            total_updates = 0
            for images, masks, bboxes in tqdm(dataloader):
                features, labels, labels_type = self.feature_extractor.forward(images, masks, bboxes)
                
                labels = labels.to(self.device)
                features = features.to(self.device)
                pairwise_similarity = compute_batch_pairwise_similarity(self.model, features, features)
                
                
                predicted = (pairwise_similarity > 0.0).float()
                labels_pairwise = labels.unsqueeze(1) == labels.unsqueeze(2)
                # test baseline
                #predicted = (labels_type.unsqueeze(1) == labels_type.unsqueeze(2)).to(predicted.device)
                #import pdb; pdb.set_trace()
                loss = self.criterion(pairwise_similarity.reshape(-1), labels_pairwise.reshape(-1).float())
                
                correct += torch.sum(torch.triu(predicted == labels_pairwise)).item()
                stat_dict[split + '_loss'] += loss.item() * predicted.numel()
                total_num += torch.triu(torch.ones_like(predicted)).sum().item()
                total_updates += predicted.numel()
                
        stat_dict[split + '_acc'] = correct / total_num
        stat_dict[split + '_loss'] = stat_dict[split + '_loss'] / total_updates
        print(f'[Eval {epoch+1}] {split} loss: {stat_dict[split + "_loss"]:.4f}, acc: {stat_dict[split + "_acc"]*100:.4f}%')
        
        return stat_dict
    
    def eval_all(self, cfg):
        stat_dict = self.eval(cfg, 0, split='test')
        
        '''
        path = os.path.join(cfg.result_dir, cfg.wandb.project_name + '_results.json')
        if os.path.exists(path):
            with open(path, "r") as f:
                my_dict = json.load(f)
        else:
            my_dict = {}
        my_dict[cfg.wandb.run_name] = stat_dict
        with open(path, "w") as f:
            json.dump(my_dict, f)
        '''

        
    def save_embeddings(self, cfg, output_dir=None):
        """Save embeddings and labels for test images to disk"""
        if output_dir is None:
            output_dir = os.path.join(cfg.dataset.data_dir, "embeddings")
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.eval()
        dataloader = self.test_dataloader
        
        with torch.no_grad():
            for batch_idx, (images, masks, bboxes) in enumerate(tqdm(dataloader, desc="Saving embeddings")):
                features, labels, _ = self.feature_extractor.forward(images, masks, bboxes)
                
                features_np = features.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                for i in range(images.shape[0]):
                    img_output_dir = os.path.join(output_dir, f"image_{batch_idx*cfg.trainer.batch_size + i}")
                    os.makedirs(img_output_dir, exist_ok=True)
                    
                    np.save(os.path.join(img_output_dir, "features.npy"), features_np[i])
                    np.save(os.path.join(img_output_dir, "labels.npy"), labels_np[i])
                    
                    img = images[i].permute(1, 2, 0).cpu().numpy()
                    plt.imsave(os.path.join(img_output_dir, "original.png"), img)


    
    def test_cross_image_similarity(self, cfg, image1_path=None, image2_path=None, masks=None, bboxes=None, metric='probe'):
        """
        Test cross-image pairwise similarity between two images.
        
        Args:
            cfg: Config object
            image1_path, image2_path: Paths to images
            masks, bboxes: Segmentation masks and bounding boxes
            metric: 'probe' (default) or 'cosine'
        
        Returns:
            similarity_matrix: [4, 4] tensor where similarity_matrix[i, j] is the average similarity
                              between object i from image1 and object j from image2
            object_mapping: dict mapping object IDs to their positions
        """
        self.model.eval()
        
        # If paths are provided, load images directly
        if image1_path and image2_path and masks is not None and bboxes is not None:
            # print("thats okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
            img1 = cv2.imread(image1_path)[:, :, ::-1]  # BGR to RGB
            img2 = cv2.imread(image2_path)[:, :, ::-1]

            images = [img1, img2]
            masks = masks
            bboxes = bboxes

        elif image1_path and image2_path:
            # print("fucking no wayyyyyyyyyyyyyyyyyyyyyyyyy")
            img1 = cv2.imread(image1_path)[:, :, ::-1]  # BGR to RGB
            img2 = cv2.imread(image2_path)[:, :, ::-1]
            
          
            # Create simple instance masks (assuming images are already arranged correctly)
            h, w = img1.shape[:2]
            mid_h, mid_w = h // 2, w // 2
            
            # Image 1: A=1 (top-left), B=2 (top-right), C=3 (bottom-left), D=4 (bottom-right)
            bbox1 = np.zeros((h, w), dtype=np.uint16)
            bbox1[0:mid_h, 0:mid_w] = 1
            bbox1[0:mid_h, mid_w:w] = 2
            bbox1[mid_h:h, 0:mid_w] = 3
            bbox1[mid_h:h, mid_w:w] = 4
            
            # Image 2: D=4 (top-left), C=3 (top-right), B=2 (bottom-left), A=1 (bottom-right)
            bbox2 = np.zeros((h, w), dtype=np.uint16)
            bbox2[0:mid_h, 0:mid_w] = 4
            bbox2[0:mid_h, mid_w:w] = 3
            bbox2[mid_h:h, 0:mid_w] = 2
            bbox2[mid_h:h, mid_w:w] = 1
            
            # Create class masks (all cars)
            mask1 = np.ones((h, w), dtype=np.uint8) * 14  # Car class
            mask2 = np.ones((h, w), dtype=np.uint8) * 14
            
            images = [img1, img2]
            masks = [mask1, mask2]
            bboxes = [bbox1, bbox2]
        else:
            # Use test dataset - assume first two images are the test images
            test_iter = iter(self.test_dataloader)
            images, masks, bboxes = next(test_iter)
            images = images[:2]
            masks = masks[:2]
            bboxes = bboxes[:2]
        
        with torch.no_grad():
            # Extract features per object for both images
            patch_features_list, patch_labels_list = self.feature_extractor.forward_per_object(
                images, masks, bboxes
            )
            
            if len(patch_features_list) < 2:
                print("Error: Could not extract features for both images")
                return None, None
            
            # Get patch features and labels
            patch_features_img1 = patch_features_list[0].to(self.device)  # [num_patches_img1, C]
            patch_features_img2 = patch_features_list[1].to(self.device)  # [num_patches_img2, C]
            patch_labels_img1 = patch_labels_list[0].to(self.device)  # [num_patches_img1]
            patch_labels_img2 = patch_labels_list[1].to(self.device)  # [num_patches_img2]
            

            print(f"Image 1: {patch_features_img1.shape[0]} patches")
            print(f"Image 2: {patch_features_img2.shape[0]} patches")
            
            # Compute patch-to-patch similarity matrix
            # patch_features_img1: [P1, C], patch_features_img2: [P2, C]
            # We want: [P1, P2] similarity matrix
            
            if metric == 'cosine':
                print("Computing Cosine Similarity...")
                # Normalize features
                feat1_norm = F.normalize(patch_features_img1, p=2, dim=1)
                feat2_norm = F.normalize(patch_features_img2, p=2, dim=1)
                # Compute cosine similarity
                patch_similarity_matrix = torch.mm(feat1_norm, feat2_norm.t()) # [P1, P2]
            else:
                print("Computing Probe-based Similarity...")
                patch_similarity = compute_batch_pairwise_similarity(
                    self.model, 
                    patch_features_img1.unsqueeze(0),  # [1, P1, C]
                    patch_features_img2.unsqueeze(0)    # [1, P2, C]
                )  # [1, P1, P2]
                patch_similarity_matrix = patch_similarity.squeeze(0)  # [P1, P2]
            
            print(f"Patch-to-patch similarity matrix shape: {patch_similarity_matrix.shape}")
            
            # Get unique object IDs from both images
            unique_objects_img1 = torch.unique(patch_labels_img1).cpu().numpy()
            unique_objects_img1 = unique_objects_img1[unique_objects_img1 > 0]  # Exclude background
            unique_objects_img2 = torch.unique(patch_labels_img2).cpu().numpy()
            unique_objects_img2 = unique_objects_img2[unique_objects_img2 > 0]  # Exclude background
            
            # Aggregate patch similarities into object-to-object similarity matrix
            # For each pair of objects (obj_i from img1, obj_j from img2),
            # average similarities between all patches labeled obj_i and all patches labeled obj_j
            ordered_obj_ids = [1, 2, 3, 4]  # Canonical order: A, B, C, D
            ordered_similarity = torch.zeros((len(ordered_obj_ids), len(ordered_obj_ids)), device=self.device)
            
            for i, obj_id_img1 in enumerate(ordered_obj_ids):
                for j, obj_id_img2 in enumerate(ordered_obj_ids):
                    # Find patches belonging to obj_id_img1 in image1
                    mask_img1 = (patch_labels_img1 == obj_id_img1)  # [P1]
                    # Find patches belonging to obj_id_img2 in image2
                    mask_img2 = (patch_labels_img2 == obj_id_img2)  # [P2]
                    
                    if mask_img1.sum() > 0 and mask_img2.sum() > 0:
                        # Extract similarities between patches of these two objects
                        obj_similarities = patch_similarity_matrix[mask_img1][:, mask_img2]  # [num_patches_obj1, num_patches_obj2]
                        # Average all patch-to-patch similarities

                        #####################change to treshold############

                        # avg_similarity = obj_similarities[obj_similarities > 0].mean().item() if (obj_similarities > 0).any() else 0.0
                        # ordered_similarity[i, j] = avg_similarity
                        #########################binary treshold##########################
                        avg_similarity = (obj_similarities > 0).float().mean().item()
                        ordered_similarity[i, j] = avg_similarity
                        #####################change to treshold############
                        # avg_similarity = obj_similarities.mean().item()
                        # ordered_similarity[i, j] = avg_similarity
                        ##########################hungerian################
                        # max_values_per_row = obj_similarities.max(dim=1).values

                        # # 2. Take the average of those maximums
                        # avg_similarity = max_values_per_row.mean().item()

                        # ordered_similarity[i, j] = avg_similarity
                        ###################################################
                        

                    else:
                        ordered_similarity[i, j] = 0.0
            
            similarity_matrix = ordered_similarity
            
            # Create mapping from object IDs to indices for visualization
            object_mapping_img1 = {int(obj_id): i for i, obj_id in enumerate(ordered_obj_ids) if obj_id in unique_objects_img1}
            object_mapping_img2 = {int(obj_id): j for j, obj_id in enumerate(ordered_obj_ids) if obj_id in unique_objects_img2}
            
            # For compatibility with visualization code, create object_ids lists
            object_ids_img1 = torch.tensor(ordered_obj_ids, dtype=torch.long)
            object_ids_img2 = torch.tensor(ordered_obj_ids, dtype=torch.long)

            # Visualize object IDs on top of the original instance masks so we can
            # inspect whether the mapping of objects is correct.
            if bboxes is not None:
                bbox1_vis = bboxes[0]
                bbox2_vis = bboxes[1]

                # Ensure numpy arrays in HxW form
                if not isinstance(bbox1_vis, np.ndarray):
                    bbox1_vis = np.array(bbox1_vis)
                if not isinstance(bbox2_vis, np.ndarray):
                    bbox2_vis = np.array(bbox2_vis)

                # If images are available, use them as background; otherwise use blank
                if "images" in locals() and len(images) >= 2:
                    img1_vis = images[0].copy()
                    img2_vis = images[1].copy()
                else:
                    h1, w1 = bbox1_vis.shape
                    h2, w2 = bbox2_vis.shape
                    img1_vis = np.zeros((h1, w1, 3), dtype=np.uint8)
                    img2_vis = np.zeros((h2, w2, 3), dtype=np.uint8)

                # Define a small color palette
                colors = [
                    (255, 0, 0),    # red
                    (0, 255, 0),    # green
                    (0, 0, 255),    # blue
                    (255, 255, 0),  # yellow
                    (255, 0, 255),  # magenta
                    (0, 255, 255),  # cyan
                ]

                overlay1 = img1_vis.copy()
                overlay2 = img2_vis.copy()

                # Color each object ID in image 1
                for k, obj_id in enumerate(object_ids_img1.tolist()):
                    color = colors[k % len(colors)]
                    mask = (bbox1_vis == int(obj_id))
                    overlay1[mask] = (
                        0.6 * overlay1[mask] + 0.4 * np.array(color, dtype=np.uint8)
                    ).astype(np.uint8)

                # Color each object ID in image 2
                for k, obj_id in enumerate(object_ids_img2.tolist()):
                    color = colors[k % len(colors)]
                    mask = (bbox2_vis == int(obj_id))
                    overlay2[mask] = (
                        0.6 * overlay2[mask] + 0.4 * np.array(color, dtype=np.uint8)
                    ).astype(np.uint8)

                # Save visualizations
                vis1_path = os.path.join(self.output_dir, "objects_img1_colored.png")
                vis2_path = os.path.join(self.output_dir, "objects_img2_colored.png")
                cv2.imwrite(vis1_path, overlay1[:, :, ::-1])  # RGB -> BGR
                cv2.imwrite(vis2_path, overlay2[:, :, ::-1])
                print(f"\nObject overlays saved to:\n  {vis1_path}\n  {vis2_path}")
            
            # similarity_matrix is already in ordered format [A, B, C, D] x [A, B, C, D]
            # It was built by aggregating patch-to-patch similarities
            print("\nCross-Image Similarity Matrix (aggregated from patch-to-patch similarities):")
            print("Rows: Image1 objects [A, B, C, D] (object IDs [1, 2, 3, 4])")
            print("Cols: Image2 objects [A, B, C, D] (but arranged as [D, C, B, A] in image)")
            sim_np = similarity_matrix.cpu().numpy()
            print(sim_np)
            print("\nExpected: High similarity on diagonal (A-A, B-B, C-C, D-D)")


            # Visualize as heatmap and save
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(sim_np, cmap="viridis")
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(["A", "B", "C", "D"])
            ax.set_yticklabels(["A", "B", "C", "D"])
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.set_xlabel("Image 2 objects")
            ax.set_ylabel("Image 1 objects")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            heatmap_path = os.path.join(self.output_dir, "cross_image_similarity_heatmap.png")
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=150)
            plt.close(fig)
            print(f"\nHeatmap saved to: {heatmap_path}")

            
            return similarity_matrix, {
                'img1_objects': object_ids_img1.tolist(),
                'img2_objects': object_ids_img2.tolist(),
                'object_mapping_img1': {k: int(v) for k, v in object_mapping_img1.items()},
                'object_mapping_img2': {k: int(v) for k, v in object_mapping_img2.items()}
            }
    





#################6 image part #####################################################################################################
    def test_multi_image_similarity(self, cfg, image_paths, masks, bboxes):
        """
        Compute similarity matrix for multiple images (N images), assuming one object per image.
        
        Args:
            image_paths: List of N image paths
            masks: List of N class masks
            bboxes: List of N instance masks
            
        Returns:
            similarity_matrix: [N, N] matrix where [i, j] is similarity between object in image i and image j
        """
        self.model.eval()
        num_images = len(image_paths)
        print(f"Computing similarity matrix for {num_images} images...")
        
        # Load all images
        images = []
        for p in image_paths:
            img = cv2.imread(p)[:, :, ::-1]  # BGR to RGB
            images.append(img)
            
        similarity_matrix = torch.zeros((num_images, num_images), device=self.device)
        
        with torch.no_grad():
            # Extract features for all images
            # forward_per_object returns list of [num_patches, C] features per image
            patch_features_list, patch_labels_list = self.feature_extractor.forward_per_object(
                images, masks, bboxes
            )
            
            if len(patch_features_list) != num_images:
                print(f"Error: Expected {num_images} feature sets, got {len(patch_features_list)}")
                return None
                
            # Compute pairwise similarity
            for i in range(num_images):
                for j in range(num_images):  # We can compute full matrix including diagonal and lower triangle
                    feat_i = patch_features_list[i].to(self.device).unsqueeze(0) # [1, Pi, C]
                    feat_j = patch_features_list[j].to(self.device).unsqueeze(0) # [1, Pj, C]
                    
                    # Compute patch-to-patch similarity
                    patch_sim = compute_batch_pairwise_similarity(self.model, feat_i, feat_j).squeeze(0) # [Pi, Pj]
                    
                    # Filter for object patches (exclude background if any retained)
                    # In segment_single_object, object has ID 1. 
                    # forward_per_object might return background patches too? 
                    # Usually it creates patches for all valid mask regions.
                    
                    lbl_i = patch_labels_list[i].to(self.device)
                    lbl_j = patch_labels_list[j].to(self.device)
                    
                    # Mask for object patches (ID > 0)
                    valid_i = lbl_i > 0
                    valid_j = lbl_j > 0
                    
                    if valid_i.sum() > 0 and valid_j.sum() > 0:
                        # Extract sub-matrix for object patches
                        obj_sim = patch_sim[valid_i][:, valid_j]
                        ####just mean
                        # similarity_matrix[i, j] = obj_sim.mean()
                        #####first positive check
                        similarity_matrix[i, j] = (obj_sim > 0).float().mean().item()
                        # similarity_matrix[i, j] = obj_sim[obj_sim > 0].mean().item() if (obj_sim > 0).any() else 0.0

                        
                    else:
                        similarity_matrix[i, j] = 0.0
                        
        print("\nMulti-Image Similarity Matrix:")
        print(similarity_matrix.cpu().numpy())
        
        return similarity_matrix   
    
#################6 image part end #####################################################################################################
