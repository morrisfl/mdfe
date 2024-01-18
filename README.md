# MDFE - Multi-Domain Feature Extraction

![model overview](readme/model_overview.png)

Intro text (e.g. abstract)

## Table of Contents


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


## IV. Training


## V. Evaluation