# FreqFusion

TPAMI 2024：Frequency-aware Feature Fusion for Dense Image Prediction

The preliminary [official implementation](https://github.com/Linwei-Chen/FreqFusion) of our TPAMI 2024 paper "[Frequency-aware Feature Fusion for Dense Image Prediction](https://www.arxiv.org/abs/2408.12879", which is also available at https://github.com/ying-fu/FreqFusion.

Interested readers are also referred to an insightful [Note]() about this work in Zhihu (TODO). 

**Abstract**

Dense image prediction tasks demand features with strong category information and precise spatial boundary details at high resolution. To achieve this, modern hierarchical models often utilize feature fusion, directly adding upsampled coarse features from deep layers and high-resolution features from lower levels. In this paper, we observe rapid variations in fused feature values within objects, resulting in intra-category inconsistency due to disturbed high-frequency features. Additionally, blurred boundaries in fused features lack accurate high frequency, leading to boundary displacement. Building upon these observations, we propose Frequency-Aware Feature Fusion (FreqFusion), integrating an Adaptive Low-Pass Filter (ALPF) generator, an offset generator, and an Adaptive High-Pass Filter (AHPF) generator. The ALPF generator predicts spatially-variant low-pass filters to attenuate high-frequency components within objects, reducing intra-class inconsistency during upsampling. The offset generator refines large inconsistent features and thin boundaries by replacing inconsistent features with more consistent ones through resampling, while the AHPF generator enhances high-frequency detailed boundary information lost during downsampling. Comprehensive visualization and quantitative analysis demonstrate that FreqFusion effectively improves feature consistency and sharpens object boundaries. Extensive experiments across various dense prediction tasks confirm its effectiveness.



## Highlight✨

- We identify two significant issues present in widely-used standard feature fusion techniques: intra-category inconsistency and boundary displacement. We also introduce feature similarity analysis to quantitatively measure these issues, which not only contributes to the development of new feature fusion methods but also has the potential to inspire advancements in related areas and beyond.
- We propose FreqFusion, which addresses category inconsistency and boundary displacement by adaptively smoothing the high-level feature with spatial-variant low-pass filters, resampling nearby category-consistent features to replace inconsistent features in the high-level feature, and enhancing the high frequency of lower-level features.
- Qualitative and quantitative results demonstrate that FreqFusion increases intra-category similarity and similarity margin, leading to a consistent and considerable improvement across various tasks, including semantic segmentation, object detection, instance segmentation, and panoptic segmentation.

<img src="../FADC/README.assets/image-20240401161246300.png" alt="图片描述" width="512">

## Code Usage

### Installation (TODO)

Our code is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). You can install mmseg by:

```
pip install mmsegmentation==0.25.0
```

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation) for more details on installation, and [dataset_prepare.md](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for information on dataset preparation. For further details on code usage, you can refer to [this](https://github.com/raoyongming/HorNet/tree/master/semantic_segmentation).

You can install mmcv-full by: 

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
```

### Results

| Method               | Backbone | Crop Size | Lr Schd | mIoU |
| -------------------- | -------- | --------- | ------- | ---- |
| SegNeXt              | MSCAN-T  | 512x512   | 160k    | 41.1 |
| SegNeXt + FreqFusion | MSCAN-T  | 512x512   | 160k    | 43.5 |

Note:

The config can be found at [here]().

(TODO)

## Acknowledgment

This code is built using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) libraries.

## Contact

If you encounter any problems or bugs, please don't hesitate to contact me at [chenlinwei@bit.edu.cn](mailto:chenlinwei@bit.edu.cn). To ensure effective assistance, please provide a brief self-introduction, including your name, affiliation, and position. If you would like more in-depth help, feel free to provide additional information such as your personal website link. I would be happy to discuss with you and offer support.