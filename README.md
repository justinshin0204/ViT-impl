# ViT-impl
This repository is an attempt to implement ViT from scratch. The goal is to understand the inner workings of BERT by building it step-by-step, rather than relying on pre-built libraries. Through this process, we aim to gain a deeper understanding of its architecture and functionality.

# ViT
The Transformer architecture has become the de facto standard for performing NLP tasks. There have also been attempts to combine it with CNN architectures in the vision domain. The authors of the ViT paper argue that it is possible to achieve good performance by applying it directly to sequences of image patches without relying on CNNs. They state that when pre-trained on large datasets and transferred to specific image recognition benchmark datasets, the **Vision Transformer (ViT)** not only delivers excellent results but also requires significantly fewer resources.

# Model Architecture & Training
Lets briefly discuss about the Model architecture of ViT and how do we train it.
![image](https://github.com/justinshin0204/ViT-impl/assets/93083019/b0696ed7-e59f-4626-84de-b7fca825d435)

I brought the image from the paper. You can see that the image has been divided into 9 patches.
We flatten each 2D patches into 1D vectors, which are then projected into a higher-dimensional space using a learned linear transformation. Additionally, 1D position embeddings are added to these vectors.


These sequences of vectors are then fed into the standard Transformer encoder, as illustrated on the right side of the picture.

Similar to BERT, ViT also adds a learnable embedding at the beginning. The embedding state z<sub>0</sub><sup>L</sup> is used as the image representation y. During both pre-training and fine-tuning, a classification head is attached to  z<sub>0</sub><sup>L</sup>. For pre-training, a single-layer MLP is used, while a simple linear layer is employed during the fine-tuning stage.

### Issues with CNNs

CNNs guide us on how to view images by initially focusing on local regions and gradually expanding to extract features. However, there's no guarantee that this is the best approach.

### Vision Transformer (ViT) Approach

ViT, on the other hand, lets the AI determine when and where to look, essentially saying, "You figure it out!" This black-box approach tends to perform better with larger datasets, as fewer inductive biases can be beneficial with more data.

Now we know how it works, let implement this.



# Implementation










