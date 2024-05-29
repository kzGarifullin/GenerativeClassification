# GenerativeClassification

## Few Shot Generative Classification

Traditional supervised classification approaches limit the scalability and training efficiency of neural networks because they require significant human effort and computational resources to partition the data.

The main goal of this research is to develop a method that reduces the need for manual annotation by training feature representations directly from unlabeled data.
## Concept

![image](https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/concept.png)

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

## Features Quality

Assessing the **separability of features** is an important step towards evaluation of models quality in learning the internal structure of dataset. To assess visually the quality of extracted features from generative models, we implemented code to project those features in 2- and 3-dimensional spaces using Uniform Manifold Approximation and Projection, UMAP. 

Features for MNIST:

|     Diffusion model       |          VAE        |          GAN        
|:-------------------------:|:-------------------:|:-------------------:
<img src="https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/MNIST-UMAP-diff.png" width="350"/> | <img src="https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/MNIST-UMAP-VAE.jpeg" width="350"/> | <img src="https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/MNIST-UMAP-GAN.png" width="350"/>

Features for CIFAR-10:

|     Diffusion model       |        VQ-VAE       |          GAN        
|:-------------------------:|:-------------------:|:-------------------:
<img src="https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/CIFAR10-UMAP-diff.gif" width="250"/> | <img src="https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/CIFAR10-UMAP-VQ-VAE.gif" width="250"/> | <img src="https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/CIFAR10-UMAP-GAN.png" width="250"/>


Features for CIFAR-100:

|     Diffusion model       |          VQ-VAE        |          GAN        
|:-------------------------:|:-------------------:|:-------------------:
<img src="https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/CIFAR100-UMAP-diff.gif" width="250"/> | <img src="https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/CIFAR100-UMAP-VQ-VAE.gif" width="250"/> | <img src="https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/CIFAR100-UMAP-GAN.gif" width="250"/>


# Results

## Results for the MNIST dataset

Comparison of three generative models for feature extraction for MNIST dataset.

![image](https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/MNIST-accuracy-comparison.png)

GAN model outperforms the baseline model on small labeled datasets, with up to 64 labeled images per class. The diffusion model and VAE also show an advantage over the baseline, with improvements up to 32 and 12 labeled images per class, respectively.


## Results for the CIFAR-10 dataset

![image](https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/cifar10_accuracy_comparison.png)

## Results for the CIFAR-100 dataset

![image](https://github.com/kzGarifullin/GenerativeClassification/blob/main/images/cifar100_accuracy_comparison.png)

## Reproducibility of diffusion experiments
All experiments were performed in Google Colab. Links to all experiments:
- [Experiment 1: diffclassification_for_mnist.ipynb](https://colab.research.google.com/drive/1wvdOL1PgP3yJP05GBEyTu1dVy-FKT3Wz?usp=sharing)
- [Experiment 2: diffusion_CIFAR_10.ipynb](https://colab.research.google.com/drive/1XzuW8fHn-Rt8UXylmINIvvWFwrITSPME?usp=sharing)
- [Experiment 3: diffusion_CIFAR_100.ipynb](https://colab.research.google.com/drive/1zYfXGbr8z0Z4Lm2SY5kvc1zq-QHnb30G?usp=sharing)

## Reproducibility of GAN experiments
All experiments were performed in Google Colab. Links to all experiments:
- [Experiment 1: GAN_mnist.ipynb](https://drive.google.com/file/d/1f80tPp4yr_jJ3O8F_awNDsdCSvZ-RsdW/view?usp=sharing)
- [Experiment 2: GAN_CIFAR_10.ipynb](https://drive.google.com/file/d/1joUih14kU0AesRifAPtWEkT72zgwyC4q/view?usp=sharing)
- [Experiment 3: GAN_CIFAR_100.ipynb](https://drive.google.com/file/d/1AvuGtvFTe6PErKvVyr1nhzpgm2HFxwiX/view?usp=sharing)


## Scripts Usage

#### Train VAE/VQ-VAE
###### MNIST VAE
```bash
    usage: train_vae_mnist.py [-h] [-d DEVICE] [-bs BATCH_SIZE] [-e EPOCHS]
                          [-lr LR] [-ld LATENT_DIM] [-pth PATH]

optional arguments:
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
  -pth PATH, --path PATH
                        Weights path
```

###### CIFAR VAE

   ```bash
usage: train_vae_cifar.py [-h] [-d DEVICE] [-bs BATCH_SIZE] [-e EPOCHS]
                          [-lr LR] [-ld LATENT_DIM] [-ct CIFAR_TYPE]
                          [-pth PATH]

optional arguments:
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
  -ct CIFAR_TYPE, --cifar_type CIFAR_TYPE
                        CIFAR10 or CIFAR100
  -pth PATH, --path PATH
                        Weights path
   ```
###### CIFAR VQ-VAE
```bash
usage: train_vqvae.py [-h] [-bs BATCH_SIZE] [-e EPOCHS] [-lr LR]
                      [-ld LATENT_DIM] [-ct CIFAR_TYPE] [-pth PATH]

optional arguments:
  -h, --help            show this help message and exit
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -lr LR, --lr LR       Learning rate
  -ld LATENT_DIM, --latent_dim LATENT_DIM
                        Laten space dimension
  -ct CIFAR_TYPE, --cifar_type CIFAR_TYPE
                        CIFAR10 or CIFAR100
  -pth PATH, --path PATH
                        Weights path
```

#### Extract Features From Different Models

###### MNIST VAE
```bash
usage: extract_features_train_smallnet_mnist.py [-h] [-p PATH] [-d DEVICE]
                                                [-s SIZE_PER_CLASS]
                                                [-e EPOCHS] [-hd HEAD]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path of weights
  -d DEVICE, --device DEVICE
                        Device for training
  -s SIZE_PER_CLASS, --size_per_class SIZE_PER_CLASS
                        Number of images per class
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -hd HEAD, --head HEAD
                        Type of head model: Lin or NonLin
```

###### CIFAR VAE
```bash
usage: extract_features_train_smallnet_cifar.py [-h] [-p PATH] [-d DEVICE]
                                                [-s SIZE_PER_CLASS]
                                                [-e EPOCHS] [-hd HEAD]
                                                [-ct CIFAR_TYPE]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path of weights
  -d DEVICE, --device DEVICE
                        Device for training
  -s SIZE_PER_CLASS, --size_per_class SIZE_PER_CLASS
                        Number of images per class
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -hd HEAD, --head HEAD
                        Type of head model: Lin or NonLin
  -ct CIFAR_TYPE, --cifar_type CIFAR_TYPE
                        CIFAR10 or CIFAR100
```

###### CIFAR VQ-VAE

```bash
usage: extract_features_train_smallnet_vqvae.py [-h] [-p PATH] [-d DEVICE]
                                                [-s SIZE_PER_CLASS]
                                                [-e EPOCHS] [-hd HEAD]
                                                [-nc NUM_CLASSES]
                                                [-ct CIFAR_TYPE]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path of weights
  -d DEVICE, --device DEVICE
                        Device for training
  -s SIZE_PER_CLASS, --size_per_class SIZE_PER_CLASS
                        Number of images per class
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -hd HEAD, --head HEAD
                        Type of head model: Lin or NonLin
  -nc NUM_CLASSES, --num_classes NUM_CLASSES
                        Num Classes
  -ct CIFAR_TYPE, --cifar_type CIFAR_TYPE
                        CIFAR10 or CIFAR100
```

## Developers
- [Kamil Garifullin](https://github.com/kzGarifullin)
- [Irina Lebedeva](https://github.com/swnirk)
- [Victoria Zinkovich](https://github.com/victoriazinkovich)
- [Ignat Melnikov](https://github.com/Minerkow)
- [Artem Alekseev](https://github.com/a063mg)


