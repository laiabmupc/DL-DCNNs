# Deep Learning: Deep Convolutional Neural Networks

#### Authors
Natàlia Muñoz Moruno, Laia Barcenilla Mañá and Núria Cardona Vilar

#### Installation
Clone the repository:
```bash
git clone <https://github.com/laiabmupc/DL-DCNNs.git>  
```

#### Overview
This repository contains the source code for the **Practical Work 1: Deep Convolutional Neural Networks** in the Deep Learning course of the Master's Degree in Artificial Intelligence (MAI). The aim of this project is to analyze the performance of several configurations of deep convolutional neural networks applied to image classification. To this end, the MAMe dataset [1] has been employed, comprising more than 37,000 256×256 images originating from museums and organized into 29 classes according to their medium. The studied architectures are a standard **CNN model with hierarchical pyramid structure** designed from scratch, an implementation of a non-standard **ResNet** network [2], and a non-standard model based on **ConvNeXtV2** [3].

#### Execution
The dataset can be downloaded from <a href="https://www.kaggle.com/datasets/ferranpares/mame-dataset/data">https://www.kaggle.com/datasets/ferranpares/mame-dataset/data</a> 

Previously to start running experiments, it is necessary to run the following scripts:
<ol>
  <li><code>divide_train_val_test.py</code>: partitions the data into training, validation and test sets.</li>
  <li><code>data_aug.py</code>: runs the data augmentation pipeline.</li>
</ol>

Then, the `main.py` file provides functions to execute the experiments.


#### References
[1] Ferran Parés, Anna Arias-Duart, Dario Garcia-Gasulla, Gema Campo-Francés, Nina Viladrich, Eduard Ayguadé, and Jesús Labarta. The mame dataset: on the relevance of high resolution and variable shape image properties. Applied Intelligence, 52(10):11703–11724, 2022.

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[3] Ronghang Hu Xinlei Chen Zhuang Liu In So Kweon Sanghyun Woo, Shoubhik Debnath and Saining Xie. Convnext v2: Co-designing and scaling convnets with masked autoencoders. arXiv preprint arXiv:2301.00808, 2023.
