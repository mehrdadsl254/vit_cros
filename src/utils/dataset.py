import sys
import os
# sys.path.append("/home/mmd/VIT/vit-object-binding/libs/dinov2")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "libs/dinov2"))
import dinov2.eval.segmentation.models

import dinov2.eval.segmentation.models.backbones.vision_transformer
import torch
from torch.utils.data import Dataset

from mmcv.runner import load_checkpoint
import os
import mmcv
import xml.etree.ElementTree as ET
import numpy as np
import dinov2.eval.segmentation.utils.colormaps as colormaps

import math
import pandas as pd
import cv2
import pickle as pkl
import sys
from tqdm import tqdm
from .segment import create_segmenter, render_segmentation, inference_segmentor, load_config_from_url
from .transforms import forward_transform, distance_point_to_bbox, inverse_transform
import matplotlib.pyplot as plt

# sys.path.append('/workspaces/003/')
sys.path.append(os.path.dirname(os.path.dirname(__file__))) # Add src to path for libs import
from libs.ADE20K.utils import utils_ade20k

activations = {}



class ADE20KSegmentation(Dataset):
    def __init__(self, root, image_set="val", index=0):
        self.root = root
        self.image_set = image_set
        self.dataset_path = os.path.join(root, "ADE20K_2021_17_01")
        self.index_file = 'index_ade20k.pkl'

        
        
        df = pd.read_csv(self.dataset_path + '/objectInfo150.csv')
        self.class_dict = {}
        names = np.array(df['Name'])
        idxs = np.array(df['Idx'])
        for idx, name in enumerate(names):
            self.class_dict[name] = idxs[idx]

        # Load dataset index
        with open(os.path.join(self.dataset_path, self.index_file), 'rb') as f:
            self.index_ade20k = pkl.load(f)
        
        obj_names = self.index_ade20k['objectnames']

        # label mapping: full label -> 150 + 1 classes
        class_mapping = [0]
        for name in obj_names:
            name = name.replace(', ', ';')
            name = name.replace(' ', ';')
            if name in self.class_dict.keys():
                obj_id = self.class_dict[name]
            elif name == 'door' or name == 'double;door':
                obj_id = 15

            elif name == 'television;receiver;television;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box':
                obj_id = 90
            elif name == 'screen;crt;screen':
                obj_id = 142
            else:
                obj_id = 0
            class_mapping.append(obj_id)
        self.class_mapping = np.array(class_mapping)

        self.image_ids = list(range(len(self.index_ade20k['filename'])))

        if image_set == 'val':
            self.image_ids = [i for i in self.image_ids if 'validation' in self.index_ade20k['folder'][i]]
        elif image_set == 'train':
            self.image_ids = [i for i in self.image_ids if 'training' in self.index_ade20k['folder'][i]]
        
    
    def __getitem__(self, idx):
        
        image_id = self.image_ids[idx]
        full_file_name = os.path.join(self.index_ade20k['folder'][image_id], self.index_ade20k['filename'][image_id])
        
        info = utils_ade20k.loadAde20K(os.path.join(self.root, full_file_name))
        
        # Load Image and Segmentation Mask
        img = cv2.imread(info['img_name'])[:, :, ::-1]  # Convert BGR to RGB
        #seg = cv2.imread(info['segm_name'])[:, :, ::-1]
        instance_mask = info['instance_mask'] # [768, 1024], 0: background, 1-n: object 
        seg = info['class_mask']

        '''
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(info['class_mask'], cmap='Greens')
        ax.axis("off")
        plt.savefig(f"/workspaces/003/src2/results/segmentation.png")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(instance_mask, cmap='Purples')
        ax.axis("off")
        plt.savefig(f"/workspaces/003/src2/results/instance.png")
        import pdb; pdb.set_trace()
        '''
        
        return img, seg, instance_mask

    def __len__(self):
        return len(self.image_ids)



class FeatureExtractor():
    def __init__(self, cfg, class_mapping):
        self.cfg = cfg
        self.class_mapping = torch.tensor(class_mapping)
        # Loading DINOV2
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[cfg.model.backbone_size]
        backbone_name = f"dinov2_{backbone_arch}"

        HEAD_SCALE_COUNT = cfg.model.head_scale# more scales: slower but better results, in (1,2,3,4,5)
        HEAD_DATASET = cfg.dataset.name # in ("ade20k", "voc2012")
        HEAD_TYPE = cfg.model.head_type # in ("ms, "linear")

        head_config_path = "/home/mmd/VIT/vit-object-binding2/weights/checkpoints/dinov2_config.py"
        head_checkpoint_url = "/home/mmd/VIT/vit-object-binding2/weights/checkpoints/dinov2_vitl14_ade20k_linear_head.pth"

        DATASET_COLORMAPS = {
            "ade20k": colormaps.ADE20K_COLORMAP,
            "voc2012": colormaps.VOC2012_COLORMAP,
        }
        
        self.backbone_model = torch.hub.load(repo_or_dir=cfg.model.model_dir, model=backbone_name, source = 'local')
        self.backbone_model.eval()
        self.backbone_model.to(cfg.trainer.device)
        
        # cfg_str = load_config_from_url(head_config_url)
        with open(head_config_path, "r") as f:
            cfg_str = f.read()
        mmcfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

        # --- FIX: Force use of standard BatchNorm (BN) instead of SyncBatchNorm (SyncBN) ---
        # SyncBN requires CUDA and typically DistributedDataParallel, causing crashes in simple inference or CPU mode.
        if mmcfg.get('norm_cfg'):
            mmcfg.norm_cfg['type'] = 'BN'
        if mmcfg.model.get('decode_head') and mmcfg.model.decode_head.get('norm_cfg'):
            mmcfg.model.decode_head.norm_cfg['type'] = 'BN'
        # -----------------------------------------------------------------------------------

        self.mmcfg = mmcfg #crop_size (512,512); stride (341, 341)
        
        self.model = create_segmenter(mmcfg, backbone_model=self.backbone_model)
        load_checkpoint(self.model, head_checkpoint_url, map_location=self.cfg.device)
        self.model.to(cfg.trainer.device)
        self.model.eval()
        
        self.colormap = DATASET_COLORMAPS[self.cfg.dataset.name]
        # Add hooks to the backbone model
        for num_layer, child in self.backbone_model.blocks.named_children():
            if int(num_layer) == self.cfg.model.num_layer[0]:
                child.register_forward_hook(get_activations(num_layer))
                num_heads = child.attn.num_heads
        self.num_heads = num_heads
        self.patch_size = self.backbone_model.patch_size

        self.num_samples = cfg.dataset.num_samples
        #import pdb; pdb.set_trace()
    
    def forward(self, imgs, masks, bboxes):
        # Input: batch data from voc2012, Output: features [B, #samples, C], labels(object_idx) [B, #samples]    
        patch_size = self.patch_size
        
        features_list = []
        labels_list = []
        labels_type_list = []
        for idx, (image, mask, bbox) in enumerate(zip(imgs, masks, bboxes)):
            activations[str(self.cfg.model.num_layer[0])] = [] # reset buffer
            
            segmentation_logits = inference_segmentor(self.model, image)[0]
            segmented_image = render_segmentation(segmentation_logits, self.cfg.dataset.name, self.colormap)
            
            activation = activations[str(self.cfg.model.num_layer[0])] #[1,1370,1024]
            if len(activation) != 2:
                continue
            patch_length = int(math.sqrt(activation[0].shape[1]))
            C = activation[0].shape[-1]

            features = torch.zeros((self.num_samples, C))
            labels = torch.zeros((self.num_samples))
            labels_type = torch.zeros((self.num_samples))
            # GROUND-TRUTH instance mask
            patch_labels = forward_transform(self.mmcfg.model.test_cfg.crop_size, self.mmcfg.model.test_cfg.stride, self.patch_size, bbox)#[[1,1,518,518], [1,1,518,518]]
            patch_types = forward_transform(self.mmcfg.model.test_cfg.crop_size, self.mmcfg.model.test_cfg.stride, self.patch_size, mask)
            half_sample_size = self.num_samples // 2
            for idx, patch_label in enumerate(patch_labels):
                patch_label = patch_label.squeeze(1)  # Shape: (1, H, W)
                N, H, W = patch_label.shape
                patch_label_unfolded = patch_label.unfold(1, 14, 14).unfold(2, 14, 14)  # Shape: (N, H//14, W//14, 14, 14)
                patch_label_unfolded = patch_label_unfolded.reshape(N, H // 14, W // 14, 14 * 14)  # Shape: (N, H//14, W//14, 196)
                patch_modes, _ = torch.mode(patch_label_unfolded, dim=-1)

                patch_type = patch_types[idx].squeeze(1)  # Shape: (1, H, W)
                patch_type_unfolded = patch_type.unfold(1, 14, 14).unfold(2, 14, 14)  # Shape: (N, H//14, W//14, 14, 14)
                patch_type_unfolded = patch_type_unfolded.reshape(N, H // 14, W // 14, 14 * 14)
                patch_modes_type, _ = torch.mode(patch_type_unfolded, dim=-1)
                
                patch_modes_type = self.class_mapping[patch_modes_type.long()]
                sample_indices = torch.randint(0, patch_length*patch_length, (half_sample_size,))

                labels[idx*half_sample_size:(idx+1)*half_sample_size] = patch_modes.reshape(-1)[sample_indices]
                features[idx*half_sample_size:(idx+1)*half_sample_size] = activation[idx][0,1:][sample_indices]
                labels_type[idx*half_sample_size:(idx+1)*half_sample_size] = patch_modes_type.reshape(-1)[sample_indices]
                
            # testbench: visualize patch and its object bbox
            '''
            import matplotlib.pyplot as plt
            patch_idx = sample_indices[0]
                
            patch_mask = torch.zeros((1, patch_length * patch_size, patch_length * patch_size))
            patch_masks = [patch_mask, patch_mask.clone()]
            window_idx = 1
            patch_y, patch_x = np.unravel_index(patch_idx, (patch_length, patch_length))
            patch_masks[window_idx][:, patch_y * patch_size: (patch_y + 1) * patch_size, patch_x * patch_size: (patch_x + 1) * patch_size] = 1

            mask_pixel = inverse_transform(self.mmcfg.model.test_cfg.crop_size, self.mmcfg.model.test_cfg.stride, image.shape, patch_masks)
            object_idx = int(labels[half_sample_size].item())
            label_idx = int(labels_type[half_sample_size].item())

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(image[:,:,::-1], alpha=0.5)  # Overlay original image
            
            where = np.where(mask_pixel[0]>0)
            y0 = np.min(where[0])
            y1 = np.max(where[0])
            x0 = np.min(where[1])
            x1 = np.max(where[1])
            ax.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], color="red", linewidth=3)
            if object_idx != 0:
                
                instance_mask = np.where(bbox == object_idx,1,0) # fence
                x0 = np.min(np.where(instance_mask)[1])
                x1 = np.max(np.where(instance_mask)[1])
                y0 = np.min(np.where(instance_mask)[0])
                y1 = np.max(np.where(instance_mask)[0])
                ax.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], color="blue", linewidth=3)
            ax.axis("off")
            plt.title(f"Object Type: {label_idx}")
            plt.savefig(f"/workspaces/003/src2/results/patch_label.png")

            
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(segmented_image, cmap='Greens')
            ax.axis("off")
            plt.savefig(f"/workspaces/003/src2/results/pred.png")
            import pdb; pdb.set_trace()
            '''
            features_list.append(features)
            labels_list.append(labels)
            labels_type_list.append(labels_type)
            
        features = torch.stack(features_list)
        labels = torch.stack(labels_list)
        labels_type = torch.stack(labels_type_list)
        return features, labels, labels_type
    
    def forward_per_object(self, imgs, masks, bboxes):
        """
        Extract features per object for cross-image similarity computation.
        Input: batch data, Output: features_per_object_list [list of [num_objects, C]], object_ids_list [list of object_ids]
        """
        patch_size = self.patch_size
        
        patch_features_list = []
        patch_labels_list = []
        
        for idx, (image, mask, bbox) in enumerate(zip(imgs, masks, bboxes)):
            activations[str(self.cfg.model.num_layer[0])] = [] # reset buffer
            
            segmentation_logits = inference_segmentor(self.model, image)[0]
            segmented_image = render_segmentation(segmentation_logits, self.cfg.dataset.name, self.colormap)
            
            activation = activations[str(self.cfg.model.num_layer[0])]
# Allow single-window activations; just require at least one
            if len(activation) == 0:
                print(f"Warning: No activation windows for image {idx}")
                continue
            
            patch_length = int(math.sqrt(activation[0].shape[1]))
            C = activation[0].shape[-1]
            
            # Get unique object IDs from bbox (instance mask)
            if isinstance(bbox, np.ndarray):
                unique_objects = np.unique(bbox)
            else:
                unique_objects = torch.unique(torch.tensor(bbox)).numpy()
            unique_objects = unique_objects[unique_objects > 0]  # Exclude background
            
            if len(unique_objects) == 0:
                print(f"Warning: No objects found in image {idx}")
                continue
            
            # GROUND-TRUTH instance mask - convert to tensor if needed
            if isinstance(bbox, np.ndarray):
                # Ensure torch-compatible dtype
                bbox_tensor = torch.from_numpy(bbox.astype(np.int32)).unsqueeze(0).unsqueeze(0)
            else:
                bbox_tensor = bbox.unsqueeze(0).unsqueeze(0) if bbox.dim() == 2 else bbox

            # forward_transform expects (H, W) mask; squeeze to 2D
            bbox_np_2d = bbox_tensor.squeeze().cpu().numpy().astype(np.int32)
            
            patch_labels = forward_transform(
                self.mmcfg.model.test_cfg.crop_size,
                self.mmcfg.model.test_cfg.stride,
                self.patch_size,
                bbox_np_2d,
            )


            
                       # Collect all patch features with their object labels
            all_patch_features = []
            all_patch_labels = []
            
            for patch_idx, patch_label in enumerate(patch_labels):
                # Some configs produce fewer activation windows than mask windows;
                # skip any mask window that doesn't have a corresponding activation.
                if patch_idx >= len(activation):
                    continue
                patch_label = patch_label.squeeze(1)  # Shape: (1, H, W)
                N, H, W = patch_label.shape
                
                # Ensure dimensions are divisible by 14
                if H % 14 != 0 or W % 14 != 0:
                    # Pad if necessary
                    pad_h = (14 - H % 14) % 14
                    pad_w = (14 - W % 14) % 14
                    patch_label = torch.nn.functional.pad(patch_label, (0, pad_w, 0, pad_h), mode='constant', value=0)
                    H, W = patch_label.shape[1], patch_label.shape[2]
                
                patch_label_unfolded = patch_label.unfold(1, 14, 14).unfold(2, 14, 14)
                patch_label_unfolded = patch_label_unfolded.reshape(N, H // 14, W // 14, 14 * 14)
                patch_modes, _ = torch.mode(patch_label_unfolded, dim=-1)  # [N, H//14, W//14]
                
                # Get all patch features from this window (excluding CLS token)
                # activation[patch_idx] shape: [1, num_patches+1, C]
                num_patches_available = activation[patch_idx].shape[1] - 1
                patch_features = activation[patch_idx][0, 1:num_patches_available+1]  # [num_patches, C]
                
                # Flatten patch labels to match patch features
                patch_labels_flat = patch_modes.reshape(-1)  # [num_patches]
                
                # Only keep patches that belong to an object (non-zero label)
                valid_mask = patch_labels_flat > 0
                if valid_mask.sum() > 0:
                    # Truncate to available patches
                    num_patches_in_window = min(len(patch_labels_flat), num_patches_available)
                    patch_features = patch_features[:num_patches_in_window]
                    patch_labels_flat = patch_labels_flat[:num_patches_in_window]
                    valid_mask = valid_mask[:num_patches_in_window]
                    
                    all_patch_features.append(patch_features[valid_mask])
                    all_patch_labels.append(patch_labels_flat[valid_mask])
            
            if len(all_patch_features) > 0:
                # Concatenate all patches from all windows
                patch_features = torch.cat(all_patch_features, dim=0)  # [total_patches, C]
                patch_labels = torch.cat(all_patch_labels, dim=0)  # [total_patches]
                
                patch_features_list.append(patch_features)
                patch_labels_list.append(patch_labels)
            else:
                print(f"Warning: No patch features extracted for image {idx}")
        
        return patch_features_list, patch_labels_list




def collate_fn(batch):
    
    images = [np.array(item[0])[:,:,::-1] for item in batch]  # Extract data from batch
    masks = [np.array(item[1]) for item in batch]  # Extract labels from batch
    bboxes = [np.array(item[2]) for item in batch]  # Extract labels from batch
    
    return images, masks, bboxes

def get_activations(name):
    def hook(model, input, output):
        if activations.get(name) is None:
            activations[name] = []
        activations[name].append(output.detach())
    return hook
if __name__ == '__main__':
    dataset = ADE20KSegmentation(root='/workspaces/003/data', image_set="train")
    print(len(dataset))
    for i in tqdm(range(len(dataset))):
        data = dataset[i]