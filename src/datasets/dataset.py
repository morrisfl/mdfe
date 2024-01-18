import os
from collections import Counter

import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm

from datasets.deep_fashion import DeepFashion
from datasets.deep_fashion2 import DeepFashion2
from datasets.fashion200k import Fashion200k
from datasets.food101 import Food101
from datasets.furniture180 import Furniture180
from datasets.hm_personalised_fashion import HuMPersonalisedFashion
from datasets.rp2k import RP2K
from datasets.shopee import Shopee
from datasets.stanford_cars import StanfordCars
from datasets.stanford_online_products import StanfordProducts
from datasets.storefronts146 import Storefronts146
from datasets.transforms import build_transforms
from datasets.food_recognition import FoodRecognition2022
from datasets.google_landmark import GLDv2
from datasets.met_dataset import METArt
from datasets.products10k import Products10k


def build_dataset(config):
    data_root = config.DATASET.root
    train_transform = build_transforms(config)
    classes = 0
    datasets = []
    for dataset_name in config.DATASET.names:
        dataset = None

        if dataset_name == "m4d-35k":
            cls = classes
            dataset = []
            p10k = Products10k(data_root, min_samples=3, max_samples=None, transform=train_transform, offset=cls)
            dataset.append(p10k)
            cls += p10k.num_classes
            gl21 = GLDv2(data_root, subset=True, transform=train_transform, offset=cls)
            dataset.append(gl21)
            cls += gl21.num_classes
            df = DeepFashion(data_root, min_samples=3, max_samples=None, num_cls=None, transform=train_transform,
                             offset=cls)
            dataset.append(df)
            cls += df.num_classes
            sc = StanfordCars(data_root, refined_cls=True, transform=train_transform, offset=cls)
            dataset.append(sc)
            cls += sc.num_classes

        elif dataset_name == "products_10k":
            dataset = Products10k(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "gldv2":
            dataset = GLDv2(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "deep_fashion":
            dataset = DeepFashion(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "met_art":
            dataset = METArt(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "shopee":
            dataset = Shopee(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "hm":
            dataset = HuMPersonalisedFashion(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "rp2k":
            dataset = RP2K(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "spo":
            dataset = StanfordProducts(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "fashion200k":
            dataset = Fashion200k(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "food_rec22":
            dataset = FoodRecognition2022(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "stanford_cars":
            dataset = StanfordCars(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "deep_fashion2":
            dataset = DeepFashion2(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "food101":
            dataset = Food101(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "furniture180":
            dataset = Furniture180(data_root, transform=train_transform, offset=classes)

        elif dataset_name == "storefronts146":
            dataset = Storefronts146(data_root, transform=train_transform, offset=classes)

        if dataset is not None:
            if isinstance(dataset, list):
                datasets.extend(dataset)
                classes = cls
            else:
                datasets.append(dataset)
                classes += dataset.num_classes
        else:
            print(f"Dataset {dataset_name} not found")

    train_dataset = ConcatDataset(datasets)

    return train_dataset, classes


def get_num_samples_per_cls(dataset, csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        num_samples_per_cls = df.to_dict()["num_samples"]
    else:
        num_samples_per_cls = Counter()
        for i in tqdm(range(len(dataset))):
            _, label = dataset[i]
            num_samples_per_cls[label] += 1
        df = pd.DataFrame.from_dict(num_samples_per_cls, orient="index", columns=["num_samples"])
        df.to_csv(csv_path)
    return num_samples_per_cls
