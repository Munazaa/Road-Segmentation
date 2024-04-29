# Road-Segmentation
Automated road segmentation is considered an essential aspect of the development and planning of cities. With the development of technologies in remote sensing and computer vision, this topic has been discussed widely from traditional to advanced methods since the 1980s: manual drawing, surveys, supervised and semi-supervised classification, and machine-learning methods.

However, automatically extracting road information from remote sensing imagery is still challenging due to the diversity in urban and rural environment structures. Considering advanced methods, road segmentation using machine learning has made significant development recently and has attained state-of-the-art performance.

## Overview
This repository presents a deep learning method for road segmentation that leverages both Very High-Resolution (VHR) remote sensing images and Volunteer Geographic Information (VGI) data. The approach aims to handle complex scenarios by jointly learning from these two diverse data sources.

## Methodology
Data Preprocessing: The initial step involves preprocessing the training dataset, which includes gathering and filtering annotated data from OpenStreetMap (OSM) using the OSMnx python library. Specific tags are selected to enhance the quality of annotations for deep learning network training.
Multi-Resolution Fusion: To incorporate more information into the network, multi-resolution images are fused to improve semantic segmentation. Two remote sensing datasets with spatial resolutions of 0.2m and 1m are utilized in training along with OSM annotations.
Deep Learning Architecture: The proposed architecture utilizes a channel attention residual U-Net model with a combined loss function to increase the precision of road extraction from remote sensing images. The channel attention technique is employed to focus the network more on roads while disregarding other objects, resulting in better segmentation.
## Results
The proposed method is compared with three state-of-the-art deep learning models: U-Net, Residual U-Net, and Attention U-Net. Experimental results demonstrate the effectiveness of the proposed approach in terms of overall accuracy and generalization capability across different areas.

## Conclusion
The proposed method showcases promising results in road segmentation, especially in handling complex road structures and achieving better connectivity of road segments. Future work may include fine-tuning approaches for even more qualitative results in diverse geographical areas.

For more details, please refer to the full paper [here](https://ieeexplore.ieee.org/abstract/document/10282859)
.
