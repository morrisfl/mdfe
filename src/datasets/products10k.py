import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


class Products10k(Dataset):
    def __init__(self, root, min_samples=3, max_samples=None, transform=None, augmentation=None, data_amount=None,
                 offset=0):
        """Parameters:
            root (str): Path to the root directory of the datasets.
            min_samples (int, optional): Minimum number of samples per class to use.
            max_samples (int, optional): Maximum number of samples per class to use.
            transform (torchvision.transforms, optional): Resize, normalize and convert image to tensor.
            augmentation (torchvision.transforms, optional): Data augmentation.
            data_amount (int, optional): Number of data to use.
            offset (int, optional): Offset of class labels.
        """
        self.root = os.path.join(root, "products-10k")
        self.min_samples = min_samples
        self.max_samples = max_samples

        self.img_dir = os.path.join(self.root, "train")
        self.df = pd.read_csv(os.path.join(self.root, "train.csv"), nrows=data_amount)
        self._sampling()

        self.class_ids = sorted(list(self.df["class"].unique()))
        self.num_classes = len(self.class_ids)

        ids_to_label = {class_id: idx + offset for idx, class_id in enumerate(self.class_ids)}
        self.df["class"] = self.df["class"].map(ids_to_label)

        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["name"]
        label = row["class"]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")

        if self.augmentation:
            img = self.augmentation(img)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.df)

    def _sampling(self):
        if self.min_samples:
            self.df = self.df.groupby("class").filter(lambda x: len(x) >= self.min_samples)
        if self.max_samples:
            for name, group in self.df.groupby("class"):
                if len(group) > self.max_samples:
                    self.df = self.df.drop(group.sample(len(group) - self.max_samples).index)


if __name__ == "__main__":
    data_root = "../../data"

    img_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.PILToTensor()])

    dataset = Products10k(data_root, transform=img_transform, data_amount=1000)
    number_cls = dataset.num_classes
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, cls in tqdm(dataloader):
        continue
