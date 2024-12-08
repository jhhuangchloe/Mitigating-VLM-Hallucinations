import torch
from torch.utils.data import Dataset
from torchvision import transforms
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
import numpy as np

import fiftyone as fo
import fiftyone.zoo as foz
from functools import partial


class OpenImagesDataset(Dataset):
    def __init__(self, num_samples=100, transform=None):
        """
        Args:
            fiftyone_dataset: A FiftyOne dataset object.
            transform: Optional PyTorch transforms to apply to the images.
        """
        self.dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="validation",
            max_samples=num_samples,
            seed=51,
            shuffle=False,
            label_types=["detections", "classifications"],
        )
        self.transform = transform

        # Store samples as a list for easy indexing
        self.samples = list(self.dataset)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample.filepath
        img = Image.open(img_path).convert("RGB")  # Load image as RGB

        # Apply transform if provided
        if self.transform:
            img = self.transform(img)

        try:
            bbox = []
            labels = []
            for data in sample.detections.detections:
                bbox.append(data.bounding_box)
                labels.append(data.label)

            bbox = torch.tensor(bbox)
            return img, labels, bbox
        except:
            return self.__getitem__(1)