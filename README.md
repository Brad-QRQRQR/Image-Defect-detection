# Corruption Detection

**Course**: Introduction to Computer Vision and its Application -- Final Project (National Central University)
**Language**: Python

## Project Overview
The project is designed to detect and localize corrupted regions within video frames. The model is trained on the "4Corruption" dataset, where 20% of the frames feature distinct visual corruptions such as Block Dropout, Tear, Mosaic Distortion, and Ghosting.

## Model Architecture
Initial iterations of the model relied on a ResNet backbone with a simple fully connected or convolutional head, which resulted in poor performance and unstable training histories. To resolve this, the proposed framework utilizes a **Fully Convolutional One-Stage (FCOS) object detector**. 

* **Backbone & Feature Pyramid Network (FPN)**: The model uses a ResNet18 backbone integrated with an FPN. The FPN extracts multi-scale features across three levels (P3, P4, and P5) with respective strides of 8, 16, and 32. This structure combines strong spatial information from shallow layers with the deep semantic understanding of the deeper layers.
* **FCOS Detection Head**: The extracted features are passed into task-specific towers consisting of 3x3 convolutions with Group Normalization and ReLU activations:
    * *Classification Head*: Predicts whether a region is foreground (corrupted) or background.
    * *Regression Head*: Calculates the continuous distances (left, top, right, bottom) from the grid point to the edges of the bounding box.
    * *Centerness Head*: Outputs a 0-1 quality score to down-weight low-quality bounding boxes that are predicted far from the center of the corrupted region.

## Loss Function & Training Pipeline
The framework is optimized using a Multi-task Loss configuration:
1.  **Classification Loss**: Sigmoid Focal Loss ($\alpha=0.25, \gamma=2.0$) to handle the severe imbalance between background and foreground.
2.  **Regression Loss**: Complete IoU (CIoU) Loss, calculated only for positive samples.
3.  **Centerness Loss**: Binary Cross Entropy (BCE) with Logits Loss, also exclusively for positive samples.

During training, the raw video data is converted to an LMDB dataset for efficient I/O operations. The pipeline applies data augmentations like color jittering and random horizontal/vertical flips. The network is trained using the Adam optimizer with a learning rate of $1e-4$, utilizes Automatic Mixed Precision (AMP), and applies a Weighted Random Sampler to further mitigate class imbalances.

## Results
The transition to the ResNet18-FPN and FCOS head yielded highly accurate and stable detection capabilities. 
* **20-Epoch Training**: The model achieved an Accuracy of **0.8730** and a validation IoU of **0.7177**.
* **50-Epoch Training (with Early Stopping)**: The best performance reached an Accuracy of **0.9120** and a validation IoU of **0.7178**.

Visualizations of the output show that the P4 classification heatmaps successfully isolate the exact locations of the defects. The model outputs high confidence scores (e.g., 0.54 to 0.79) when tightly bounding corrupted regions, while correctly assigning low confidence scores (e.g., ~0.13) and suppressing bounding boxes on clean, normal images.