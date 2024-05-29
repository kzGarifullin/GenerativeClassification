# GenerativeClassification

## Few Shot Generative Classification

Traditional supervised classification approaches limit the scalability and training efficiency of neural networks because they require significant human effort and computational resources to partition the data.

The main goal of this research is to develop a method that reduces the need for manual annotation by training feature representations directly from unlabeled data.
## Concept

![image](https://github.com/David-cripto/DiffClassification/assets/78556639/cbe5f13e-c6f2-4021-bf86-dca3c87d5d6c)

Given a scenario where we possess a small labeled dataset alongside a larger unlabeled dataset, we can approach classification through the following steps:

- Train a generative model on the unlabeled dataset to learn the underlying data distribution.
- Utilize the trained generative model to extract more representative features for the labeled images, effectively enriching the feature space.
- Train a small neural network using the enriched features to make predictions for the corresponding labels.

This approach leverages the generative model to enable the small neural network to make accurate predictions despite the limited labeled data.


## Optimal time selection for diffusion model
In the framework of the diffusion model for feature aggregation, the choice of the optimal diffusion time step parameter becomes paramount in determining the temporal influence of features.

MNIST             |  CIFAR-10             |  CIFAR-100
:-------------------------:|:-------------------------:|:-------------------------:
![image](https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/MNIST-t-opt.png) | ![image](https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/CIFAR-10-t-opt.png) | ![image](https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/CIFAR-100-t-opt.png)

As evident from the plots, the optimal step values for MNIST and CIFAR-10 are $100$ and $50$, respectively. Consequently, we set these timesteps as constants for subsequent experiments.

## Generation Quality

The efficiency of a model's generation can be directly related to its feature representation. To assess models' performance on datasets, we employed generation metrics. Specifically, we utilized the Frechet Inception Distance (FID) and Density Coverage metrics. FID measures the disparity between the distributions of generated and actual data samples in feature space, with lower FID scores indicating better performance. This metric accounts for both the accuracy and diversity of generated outputs.
On the other hand, Density Coverage assesses a model's capability to recreate the underlying data distribution. 

VAE generation             |  Diffusion generation
:-------------------------:|:-------------------------:
![image](https://github.com/David-cripto/DiffClassification/blob/kzGarifullin-patch-1/assets/MNIST/mnist-generation-cifar.PNG) | ![image](https://github.com/David-cripto/DiffClassification/blob/kzGarifullin-patch-1/assets/MNIST/mnist-generation-diff.PNG)


## Features Quality

Assessing the **separability of features** is an important step towards evaluation of models quality in learning the internal structure of dataset. To assess visually the quality of extracted features from generative models, we implemented code to project those features in 2- and 3-dimensional spaces using Uniform Manifold Approximation and Projection, UMAP. 

Features for diffusion model:

| MNIST dataset |  CIFAR-10 dataset |
|:-------------------------:|:-------------------:|
![MNIST](https://github.com/David-cripto/DiffClassification/blob/kzGarifullin-patch-1/assets/MNIST/diff_mnist_umap.png) | ![MNIST](https://github.com/David-cripto/DiffClassification/blob/kzGarifullin-patch-1/assets/MNIST/diff_cifar_umap.png) | 

Features for Variational Autoencoder:

| MNIST dataset |  CIFAR-10 dataset |
|:-------------------------:|:-------------------:|
![image](https://github.com/David-cripto/DiffClassification/blob/kzGarifullin-patch-1/assets/MNIST/mnist-features.png) | ![image](https://github.com/David-cripto/DiffClassification/blob/kzGarifullin-patch-1/assets/MNIST/cifar-features.png)


# Results

## MNIST Training Results

Training linear (Linear) and nonlinear (Linear+ReLU+Linear) models on features from generative models

![image](https://github.com/David-cripto/DiffClassification/assets/78556639/9397cc93-c248-461f-aace-6bbab676224d)

![image](https://github.com/David-cripto/DiffClassification/assets/78556639/6891d2ed-2740-4f14-ac96-df0d20d093c2)


## Models Comparison

![image](https://github.com/David-cripto/DiffClassification/assets/78556639/7af1b8f4-0df9-485e-b40a-d796c3ed97fb)

## CIFAR-10 Training Results

Training nonlinear (Linear+ReLU+Linear) model (since it was the best on MNIST) on features from generative models
![image](https://github.com/David-cripto/DiffClassification/assets/78556639/f77f8155-c96a-40db-8d07-d094e0d458dd)

## Models comparison

![image](https://github.com/David-cripto/DiffClassification/assets/78556639/8ea8a96e-5c4a-432f-8318-522985b39130)

## Scripts Usage

#### VAE

   ```bash
  extract_features_train_smallnet_cifar.py [-h] [-p PATH] [-d DEVICE] [-s SIZE_PER_CLASS] [-e EPOCHS]
options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path of weights
  -d DEVICE, --device DEVICE
                        Device for training
  -s SIZE_PER_CLASS, --size_per_class SIZE_PER_CLASS
                        Number of images per class
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
   ```
```bash
extract_features_train_smallnet_mnist.py [-h] [-p PATH] [-d DEVICE] [-s SIZE_PER_CLASS] [-e EPOCHS]

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path of weights
  -d DEVICE, --device DEVICE
                        Device for training
  -s SIZE_PER_CLASS, --size_per_class SIZE_PER_CLASS
                        Number of images per class
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
```

 ```bash
train_vae_cifar.py [-h] [-d DEVICE] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LR] [-ld LATENT_DIM]

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Device for training
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -lr LR, --lr LR       Learning rate
  -ld LATENT_DIM, --latent_dim LATENT_DIM
                        Laten space dimension
 ```

 ```bash
train_vae_mnist.py [-h] [-d DEVICE] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LR] [-ld LATENT_DIM]

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Device for training
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -lr LR, --lr LR       Learning rate
  -ld LATENT_DIM, --latent_dim LATENT_DIM
                        Laten space dimension
 ```


#### Diffusion

mnist_main.ipynb - loading pretrained weights of DDPM and feature extraction on MNIST dataset  

CIFAR_10_Diffusion_new.ipynb - training DDPM and feature extraction on CIFAR-10 dataset 

supervised.ipynb - training ResNet18 for comparising with small net trained on features extracted from DDPM on different train set sizes 

## Pretrained weights

Pretrained VAE models: https://drive.google.com/drive/folders/1UrXq-gdDHtKQBUMxOc-oiawpqYo2ozga?usp=share_link


Pretrained diffusion model for CIFAR-10: https://drive.google.com/file/d/1ICLWfz3Wu8cVQhJUBOcxlEuGsw6zbaFL/view?usp=drive_link

Pretrained diffusion model for MNIST: https://drive.google.com/uc?id=1fSPB08M6aBNmhjRgSn3qpdq5hXl1Xhao




## Developers
- [Kamil Garifullin](https://github.com/kzGarifullin)
- [Irina Lebedeva](https://github.com/swnirk)
- [Victoria Zinkovich](https://github.com/victoriazinkovich)
- [Ignat Melnikov](https://github.com/Minerkow)
- [David Li](https://github.com/David-cripto)


