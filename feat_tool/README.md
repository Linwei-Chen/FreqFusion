# Feature Similarity Analysis for Dense Image Prediction

## Introduction

This repository contains the code for feature similarity analysis as described in the paper "Frequency-aware Feature Fusion for Dense Image Prediction" published in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 2024. The code is designed to analyze the feature similarity in dense image prediction tasks, focusing on intra-category consistency and boundary displacement.

## Code Overview

The provided code includes the following key components:

1. **Feature Similarity Analysis with Edge Detection (feat_sim_analysis_with_edge)**:
   - Extends the feature similarity analysis by incorporating edge detection.
   - Computes the similarity of features along the boundaries (edges) of objects.
   - Helps in evaluating the boundary displacement issue.
2. **Feature Fusion Analysis (feat_fusion_analysis)**:
   - Analyzes the feature fusion process by combining features from different layers.
   - Evaluates the impact of feature fusion on intra-class similarity and boundary displacement.

## Getting Started

### Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.6 or higher
- NumPy
- Matplotlib
- OpenCV
- PyTorch
- SciPy

You can install the required packages using the following command:

```bash
pip install numpy matplotlib opencv-python torch scipy
```

### Dataset Preparation

The code is designed to work with the Cityscapes dataset. Ensure you have the dataset downloaded and organized in the following structure:

```
cityscapes/
  leftImg8bit/
    val/
      frankfurt/
        frankfurt_000001_060135_leftImg8bit.png
        ...
  gtFine/
    val/
      frankfurt/
        frankfurt_000001_060135_labelIds.png
        ...
```

### Code Usage

Before using the code, please make sure to save the features of each network layer as `.npy` files in the `CxHxW` format, where `C` is the number of channels, `H` is the height, and `W` is the width. The filenames should match the names of the input images so that the code can correctly align the features and images. Features from different network layers should be stored in separate directories to distinguish them. When using the code, you can set the `feat_path` parameter to the path of the directory where the layer features are saved. For example:

Python复制

```python
feat_sim_analysis_with_edge('/path/to/layer1_features', edge_k=0, channel_clip=True, gt_size=(256, 512), debug=False, inter_mode='nearest', num=50)
```

In this example, `'/path/to/layer1_features'` is the directory path where the features of the first layer are saved.

#### Feature Similarity Analysis

To run the feature similarity analysis, use the following command:

```bash
python tpami2024-feature_similarity_analysis_released.py
```

By default, the code will analyze the feature similarity for the first 10 images in the validation set. You can adjust the number of images by modifying the `num` parameter in the `feat_sim_analysis_with_edge` function.

#### Feature Fusion Analysis

To run the feature fusion analysis, modify the `feat_path_list` parameter in the `feat_fusion_analysis` function to point to the feature paths you want to analyze. Then, run the script:

bash复制

```bash
python tpami2024-feature_similarity_analysis_released.py
```

## Code Explanation

### Feature Similarity Analysis with Edge Detection

The `feat_sim_analysis_with_edge` function extends the `feat_sim_analysis` function by incorporating edge detection:

1. **Edge Detection**:
   - Detects the edges in the ground truth labels using the Laplacian operator.
   - Dilates the edges to create a boundary region.
2. **Compute Intra-Class and Inter-Class Similarity**:
   - Computes the similarity between the features along the edges and the category center.
   - Computes the similarity between different classes.
3. **Error Analysis**:
   - Analyzes the error rates for each class.
   - Computes the overall accuracy.

### Feature Fusion Analysis

The `feat_fusion_analysis` function analyzes the feature fusion process:

1. **Load Features from Multiple Layers**:
   - Loads the feature maps from multiple layers.
   - Concatenates the feature maps to form a combined feature map.
2. **Compute Intra-Class and Inter-Class Similarity**:
   - Computes the similarity between the combined features and the category center.
   - Computes the similarity between different classes.
3. **Error Analysis**:
   - Analyzes the error rates for each class.
   - Computes the overall accuracy.

## Results

The code will output the following results:

- **Intra-Class Similarity**: The average similarity within each class.
- **Inter-Class Similarity**: The average similarity between different classes.
- **Error Rates**: The error rates for each class.

## License

Please cite the paper if you use this code in your research.

```bibtex
@article{chen2024frequency,
  author={Chen, Linwei and Fu, Ying and Gu, Lin and Yan, Chenggang and Harada, Tatsuya and Huang, Gao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Frequency-aware Feature Fusion for Dense Image Prediction}, 
  year={2024},
  volume={46},
  number={12},
  pages={10763-10780},
  doi={10.1109/TPAMI.2024.3449959}
}
```