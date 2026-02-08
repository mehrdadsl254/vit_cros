# Cross-Image Similarity Test

This test evaluates the model's ability to match objects across two images where the same objects appear in different arrangements.

## Test Setup

The test uses two images:
- **Image 1**: Objects arranged as A, B / C, D (top-left, top-right / bottom-left, bottom-right)
- **Image 2**: Same objects rearranged as D, C / B, A

The goal is to compute a similarity matrix where:
- Rows represent objects from Image 1: [A, B, C, D]
- Columns represent objects from Image 2: [A, B, C, D] (but arranged as [D, C, B, A] in the image)

The expected result is a diagonal matrix where:
- A from Image 1 matches A from Image 2
- B from Image 1 matches B from Image 2
- C from Image 1 matches C from Image 2
- D from Image 1 matches D from Image 2

## Files Created

1. **`src/utils/create_test_data.py`**: Script to create ADE20K format test data from two images
2. **`src/utils/dataset.py`**: Added `forward_per_object()` method to extract features per object
3. **`src/trainer_ce.py`**: Added `test_cross_image_similarity()` method
4. **`src/test_cross_image_similarity.py`**: Complete test script

## Usage

### Option 1: Using existing images (1.png and 2.png)

If you have two images named `1.png` and `2.png` in the project root:

```bash
cd src
python test_cross_image_similarity.py --config cfgs/config.yaml --checkpoint path/to/checkpoint.pth
```

### Option 2: The script will create test images automatically

If the images don't exist, the script will create simple colored rectangles:
- Image 1: Red (A), Green (B), Blue (C), Yellow (D)
- Image 2: Yellow (D), Blue (C), Green (B), Red (A)

### Option 3: Using actual car images

To use actual car images from Stanford Cars dataset or similar:

1. Prepare two images with 4 car objects arranged as specified
2. Place them as `1.png` and `2.png` in the project root
3. Run the test script

## How It Works

1. **Data Creation**: The script creates ADE20K format data with:
   - Instance masks (object IDs: 1=A, 2=B, 3=C, 4=D)
   - Segmentation masks (class: car)
   - JSON annotations

2. **Feature Extraction**: For each image:
   - Extract DINOv2 features using the FeatureExtractor
   - Group patches by object ID
   - Average features per object

3. **Similarity Computation**: 
   - Compute pairwise similarity between all objects from Image 1 and Image 2
   - Create a 4x4 similarity matrix
   - Reorder to match expected object order

4. **Evaluation**:
   - Check if diagonal values (correct matches) are higher than off-diagonal values
   - Print similarity matrix
   - Save results to file

## Expected Output

```
Cross-Image Similarity Matrix:
Rows: Image1 objects [A, B, C, D] (object IDs [1, 2, 3, 4])
Cols: Image2 objects [A, B, C, D] (but arranged as [D, C, B, A] in image)
[[0.85  0.12  0.08  0.15]   # A matches A (high), others low
 [0.10  0.82  0.15  0.11]   # B matches B (high), others low
 [0.09  0.14  0.88  0.10]   # C matches C (high), others low
 [0.12  0.09  0.11  0.86]]  # D matches D (high), others low

Expected: High similarity on diagonal (A-A, B-B, C-C, D-D)
✓ SUCCESS: All correct matches have higher similarity than incorrect matches!
```

## Integration with Existing Code

The test integrates seamlessly with the existing codebase:

- Uses the same `FeatureExtractor` class
- Uses the same `compute_batch_pairwise_similarity` function
- Compatible with existing model checkpoints
- Follows ADE20K data format conventions

## Notes

- The test assumes objects are arranged in a 2x2 grid (4 quadrants)
- Object IDs must be 1, 2, 3, 4 for A, B, C, D respectively
- The test works with any model checkpoint trained with the existing training pipeline

