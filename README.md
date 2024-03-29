# Symbolic Temporal Pooling
Source code for the paper "A Symbolic Temporal Pooling method for Video-based
Person Re-Identification". 

# Introduction

In video-based person re-identification, both the spatial and temporal features are known to provide orthogonal cues to discriminative video-level representations. In order to obtain feature representations, the frame-level features are typically aggregated using max/avg pooling, at different points of the classification frameworks. However, obtaining such compact representations of the input data leads invariably to loose information, which can be particularly hazardous in case of poor separability between the different classes (IDs). To alleviate this problem, this work introduces a symbolic temporal pooling method, where frame-level features are represented in a distribution-valued symbolic form, yielding from fitting an Empirical Cumulative Distribution Function (ECDF) to each feature. Moreover, considering that the original triplet loss cannot be applied directly to this kind of representations, we introduce a symbolic triplet loss function that infers the loss between two symbolic objects.  

![Alt text](/Images/Architecture.png?raw=true "Title")

# Datasets
We evaluated the performance of the proposed method on four well known data sets (MARS, iLIDS-VID, PRID2011 and P-DESTRE). Please follow the instructions in [https://github.com/KaiyangZhou/deep-person-reid] to prepare the data for MARS, iLIDS-VID, PRID2011 datasets.
For P-DESTRE dataset, extract the bounding box for each person and save it in a single folder name with "PID". Use the evalaution protocol from [ http://p-destre.di.ubi.pt/]

# Credits
The source code is built upon the github repositories [https://github.com/jiyanggao/Video-Person-ReID]
