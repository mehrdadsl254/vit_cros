# SAM (Segment Anything Model) Installation Guide

This script now uses **SAM (Segment Anything Model)** by Meta for the best segmentation quality. SAM is one of the most powerful segmentation models available.

## Installation

### Option 1: Install SAM via pip

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Option 2: Install from source

```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

## Download SAM Checkpoints

Download the SAM model checkpoint from:
https://github.com/facebookresearch/segment-anything#model-checkpoints

Recommended models:
- **sam_vit_h_4b8939.pth** (Best quality, ~2.4GB) - Recommended
- **sam_vit_l_0b3195.pth** (Good quality, ~1.2GB)
- **sam_vit_b_01ec64.pth** (Fast, ~375MB)

Place the checkpoint file in one of these locations:
- `checkpoints/sam_vit_h_4b8939.pth`
- `~/.cache/sam/sam_vit_h_4b8939.pth`
- Or specify path with `--sam_checkpoint` argument

## Usage

### With SAM (Best Quality)

```bash
cd src/utils
python create_test_data.py \
    --image1 /path/to/image1.png \
    --image2 /path/to/image2.png \
    --output_dir ../../ \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth
```

### Without SAM (Fallback)

If SAM is not available, the script will automatically fall back to OpenCV-based segmentation:

```bash
python create_test_data.py \
    --image1 /path/to/image1.png \
    --image2 /path/to/image2.png \
    --output_dir ../../ \
    --no_sam
```

## How SAM Works

SAM uses point prompts to segment objects:
1. For each quadrant, it uses the center point as a prompt
2. SAM generates multiple mask proposals
3. The best mask (highest score) is selected
4. The mask is refined to remove noise and fill holes

This provides much more accurate object boundaries compared to traditional methods like GrabCut.

## Requirements

- Python 3.8+
- PyTorch (for SAM)
- OpenCV
- NumPy

Optional but recommended:
- CUDA (for GPU acceleration with SAM)

