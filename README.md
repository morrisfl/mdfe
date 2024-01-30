# MDFE - Multi-Domain Feature Extraction

![model overview](readme/model_overview.svg)
**Figure:** *Overview of the proposed multi-domain image embedding model. The model consists of a visual-semantic foundation
model as backbone with an attached projection layer. The model was trained on a custom curated multi-domain training dataset (M4D-35k), 
using a margin-based softmax loss.*

This repository is related to the research conducted for my master thesis entitled "**Efficient and Discriminative Image 
Feature Extraction for Multi-Domain Image Retrieval**". An executive summery of the thesis can be found [here](readme/executive_summary.pdf). 
For those interested in a comprehensive review of the entire work, please feel free to contact me to obtain a copy of the 
full thesis report.

### Abstract
The prevalence of image capture devices has led to the growth of digital image collections, requiring advanced retrieval systems. 
Current methods are often limited by their domain specificity, struggle with out-of-domain images, and lack of generalization. 
This study addresses these limitations by focusing on multi-domain feature extraction. The goal entails in developing an 
efficient multi-domain image encoder for fine-grained retrieval while overcoming computational constraints. Therefore, a 
multi-domain training dataset, called *M4D-35k*, was curated, allowing for resource-efficient training. Dataset 
curation involved selecting from 15 datasets and optimizing their overall size in terms of samples and classes used. 
Additionally, the effectiveness of various visual-semantic foundation models and margin-based softmax loss were evaluated 
to assess their suitability for multi-domain feature extraction. Among the loss functions investigated, a proprietary approach 
was developed that refers to *CLAM* (**cl**ass distribution aware additive **a**ngular **m**argin loss). Even with computational 
limitations, a close to SOTA result was achieved on the [Google Universal Image Embedding Challenge](https://www.kaggle.com/competitions/google-universal-image-embedding) 
(GUIEC) evaluation dataset. Linear probing of the embedding model alone resulted in a mMP@5 score of 0.722. The total 
number of model parameters and the number of trainable parameters were reduced by 32% and 289 times, respectively. Despite 
the smaller model and without end-to-end fine-tuning, it trailed the GUIEC leaderboard by only 0.8%, surpassing 2nd place 
and closely behind 1st. It also outperformed the top-ranked method with similar computational prerequisites by 3.6%.

### Results
|                                              GUIEC rank                                              | Method         | # total model params | # trainable params | mMP@5 |
|:----------------------------------------------------------------------------------------------------:|----------------|:--------------------:|:------------------:|:-----:|
| [1st place](https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/359316)  | fine-tuning    |         661M         |        661M        | 0.730 |
| [2nd place](https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/359525)  | fine-tuning    |         667M         |        667M        | 0.711 |
| [5th place](https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/359161)  | linear probing |         633M         |        1.1M        | 0.686 |
| [10th place](https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/359271) | linear probing |        1,045M        |       22.0M        | 0.675 |
|                                             Own approach                                             | linear probing |         431M         |        2.3M        | 0.722 |

**Table:** *Comparison of the proposed approach with the top-ranked methods on the GUIEC evaluation dataset. It improves 
the total model parameters at inference by 32% compared to the leanest approach (5th place), reduces the number of trainable 
parameters by 289x compared to the fine-tuning approaches (1st and 2nd place), and achieves a performance close to SOTA, 
surpassing 2nd place and just behind 1st place.*

## Table of Contents
- [I. Setup](#i-setup)
- [II. Data Preparation](#ii-data-preparation)
- [III. Embedding Model](#iii-embedding-model)
- [IV. Training](#iv-training)
- [V. Evaluation](#v-evaluation)

## I. Setup
Here, we describe a step-by-step guide to setup and install dependencies on a UNIX-based system, such as Ubuntu, using 
`conda` as package manager. If `conda` is not available, alternative package managers such as `venv` can be used.

#### 1. Create a virtual environment
```
conda create -n env_mdfe python=3.8
conda activate env_mdfe
```
#### 2. Clone the repository
```
git clone git@github.com:morrisfl/mdfe.git
```
#### 3. Install pytorch
Depending on your system and compute requirements, you may need to change the command below. See [pytorch.org](https://pytorch.org/get-started/locally/) 
for more details. In order to submit the embedding models to the 2022 [Google Universal Image Embedding Challenge](https://www.kaggle.com/competitions/google-universal-image-embedding), 
PyTorch 1.11.0 is required.
```
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```
#### 4. Install the repository with all dependencies
```
cd mdfe
python -m pip install .
```
If you want to make changes to the code, you can install the repository in editable mode:
```
python -m pip install -e .
```
#### 5. Setup Google Drive access (optional)
In order to automatically upload checkpoints to Google Drive, you need to create a Google Drive API key. 
Setup instructions can be found [here](src/utils/google_drive.md).

## II. Data Preparation
In the process of fine-tuning/linear probing the embedding models, the following dataset can be used:

| Dataset                                                                                                           |        Domain         |    Config key    | Note                                                                             |
|-------------------------------------------------------------------------------------------------------------------|:---------------------:|:----------------:|----------------------------------------------------------------------------------|
| [Products-10k](https://products-10k.github.io)                                                                    |    Packaged goods     |  `products_10k`  |                                                                                  |
 | [Google Landmarks v2](https://www.kaggle.com/c/landmark-recognition-2021/data)                                    |       Landmarks       |     `gldv2`      | cleaned subset of GLDv2 is used.                                                 |
| [DeepFashion (Consumer to Shop)](https://www.kaggle.com/datasets/sangamman/deepfashion-consumer-to-shop-training) | Apparel & Accessories |  `deep_fashion`  |                                                                                  |
| [MET Artwork](http://cmp.felk.cvut.cz/met/)                                                                       |        Artwork        |    `met_art`     |                                                                                  |
 | [Shopee](https://www.kaggle.com/competitions/shopee-product-matching/data)                                        |    Packaged goods     |     `shopee`     |                                                                                  |
 | [H&M Personalized Fashion](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) | Apparel & Accessories |       `hm`       |                                                                                  |
 | [RP2K](https://www.pinlandata.com/rp2k_dataset/)                                                                  |    Packaged goods     |      `rp2k`      |                                                                                  |
 | [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)                                     |    Packaged goods     |      `sop`       |                                                                                  |
 | [Fashion200k](https://github.com/xthan/fashion-200k)                                                              | Apparel & Accessories |  `fashion200k`   | annotations in csv format (see `data/fashion200k_train.csv`)                     |
 | [Food Recognition 2022](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022#datasets)              |     Food & Dishes     |   `food_rec22`   | dataset must to bee preprocessed according to `data/food_rec22_preprocess.py`    |
 | [Stanford Cars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)                              |         Cars          | `stanford_cars`  | annotations in csv format (see `data/sc_train.csv`)                              |
 | [DeepFashion2](https://github.com/switchablenorms/DeepFashion2)                                                   | Apparel & Accessories | `deep_fashion2`  | dataset must to bee preprocessed according to `data/deep_fashion2_preprocess.py` |
 | [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)                                            |     Food & Dishes     |    `food101`     | test images are used for training.                                               |
 | [Furniture 180](https://www.kaggle.com/datasets/andreybeyn/qudata-gembed-furniture-180)                           |       Furniture       |  `furniture180`  | annotations in csv format (see `data/furniture180_train.csv`)                    |
 | [Storefornts 146](https://www.kaggle.com/datasets/kerrit/storefront-146)                                          |      Storefronts      | `storefronts146` | annotations in csv format (see `data/storefronts146_train.csv`)                  |

Download the datasets and place them in a `<data_dir>` of your choice. The directory structure should look as follows:
```
<data_dir>/
├── products-10k/
│   ├── train
│   └── train.csv
├── google_landmark_recognition_2021/
│   ├── train
│   ├── train.csv
│   └── gldv2_train_cls10000.csv
├── deepfashion/
│   ├── train
│   └── deepfashion_train.json
├── met_dataset/
│   ├── MET
│   └── ground_truth/MET_database.json
├── shopee/
│   ├── train_imahes
│   └── train.csv
├── hm_personalized_fashion/
│   ├── images
│   └── articles.csv
├── rp2k/
│   ├── train
│   └── train.csv
├── stanford_online_products/
│   ├── <img_dirs>
│   └── Ebay_train.txt
├── fashion200k/
│   ├── women
│   └── fashion200k_train.csv
├── fr22_train_v2/
│   ├── images
│   ├── preprocessed_imgs
│   ├── annotations.json
│   └── train.csv
├── stanford_cars/
│   ├── cars_train
│   ├── sc_train.csv
│   └── sc_refined_train.csv
├── deep_fashion2/
│   ├── image
│   ├── annos
│   ├── preprocessed_imgs
│   └── train.csv
├── food-101/
│   ├──images
│   └── meta/test.json
├── furniture_180/
│   ├── <img_dirs>
│   └── furniture180_train.csv
└── storefronts_146/
    ├── <img_dirs>
    └── storefronts146_train.csv
```

The above-mentioned datasets can be included into the training process by adding the corresponding `config key` to the
`DATASET.names` parameter in the configuration file in `configs/`.

#### *M4D-35k*
The *M4D-35k* dataset is a custom curated multi-domain training dataset. It was created for resource-efficient training of 
multi-domain image embeddings. The curation process involved dataset selection data sampling (optimize data size) by 
maximizing the performance on the GUIEC evaluation dataset. *M4D-35k* consists of the following datasets:

- Products-10k
- 10k classes from Google Landmarks v2. The corresponding landmark annotations are available in `data/gldv2_train_cls10000.csv`.
- DeepFashion (Consumer to Shop)
- A refined version of Stanford Cars. The corresponding annotations are available in `data/sc_refined_train.csv`.

To use *M4D-35k* for training, add `m4d_35k` to the `DATASET.names` parameter in the configuration file in `configs/`.

## III. Embedding Model

| Foundation Model | Encoder architecture |     `type`      |   `model_name`   |    `weights`     |
|------------------|----------------------|:---------------:|:----------------:|:----------------:|
| [CLIP]()         | ViT                  |     `clip`      | see [OpenCLIP]() | see [OpenCLIP]() |
 | [CLIP]()         | ConvNeXt             | `clip_convnext` | see [OpenCLIP]() | see [OpenCLIP]() |
 | [CLIPA]()        | ViT                  |    `clipav2`    | see [OpenCLIP]() | see [OpenCLIP]() | 
 | [EVA-CLIP]()     | ViT                  |     `eva02`     |   see [timm]()   |        -         | 
 | [MetaCLIP]()     | ViT                  |   `meta-clip`   | see [OpenCLIP]() | see [OpenCLIP]() | 
 | [SigLIP]()       | ViT                  |    `siglip`     |   see [timm]()   |        -         |
 | [DINOv2]()       | ViT                  |    `dinov2`     |   see [timm]()   |        -         |
 | [SAM]()          | ViT                  |      `sam`      |   see [timm]()   |        -         |




## IV. Training


## V. Evaluation