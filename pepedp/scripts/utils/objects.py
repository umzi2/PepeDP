import os
import shutil

import torch
from torch.utils.data import DataLoader

from .module import ImageDataset
from tqdm import tqdm


class Threshold:
    def __init__(self, name, threshold):
        self.name = name
        self.threshold = threshold

    def __repr__(self):
        return f"Threshold(Name = {self.name}, Threshold = {self.threshold}\n)"


class ThresholdList:
    def __init__(self):
        self.mass = []

    def append(self, threshold):
        self.mass.append(threshold)

    def extend(self, threshold_list):
        self.mass.extend(threshold_list)

    def sort(self, reverse: bool = False):
        self.mass.sort(key=lambda item: item.threshold, reverse=reverse)

    def __iter__(self):
        return iter(self.mass)

    def __getitem__(self, index):
        return self.mass[index]

    def __len__(self):
        return len(self.mass)


class IQANode:
    def __init__(
        self,
        img_dir,
        batch_size: int = 8,
        threshold: float = 0.5,
        median_threshold=0,
        move_folder: str | None = None,
        transform=None,
        reverse=False,
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dataset = ImageDataset(img_dir, self.device, transform)
        self.data_loader = DataLoader(dataset, batch_size=batch_size)
        self.img_dir: str = img_dir
        self.threshold = threshold
        self.reverse = reverse
        self.move_folder = move_folder
        if move_folder is not None:
            import os

            os.makedirs(move_folder, exist_ok=True)
        if median_threshold:
            self.median_threshold = median_threshold
            self.threshold_list = ThresholdList()
        else:
            self.threshold_list = None

    def forward(self, images):
        raise NotImplementedError(
            "The forward method must be implemented in a subclass"
        )

    @torch.inference_mode()
    def __call__(self):
        for images, filenames in tqdm(self.data_loader):
            iqa = self.forward(images)
            for index in range(iqa.shape[0]):
                file_name = filenames[index]
                iqa_value = iqa[index]

                if self.threshold_list is None:
                    if iqa[index] > self.threshold and self.move_folder:
                        shutil.move(  # shutil.clone(
                            os.path.join(self.img_dir, file_name),
                            os.path.join(self.move_folder, file_name),
                        )
                    elif iqa[index] < self.threshold and not self.move_folder:
                        os.remove(os.path.join(self.img_dir, file_name))
                else:
                    if (iqa[index] > self.threshold and not self.reverse) or (
                        iqa[index] < self.threshold and self.reverse
                    ):
                        self.threshold_list.append(
                            Threshold(name=file_name, threshold=float(iqa_value))
                        )
                    else:
                        if not self.move_folder:
                            os.remove(os.path.join(self.img_dir, file_name))
        if self.threshold_list:
            self.threshold_list.sort(self.reverse)
            clip_index = int(len(self.threshold_list) * self.median_threshold)
            if self.move_folder:
                for threshold in self.threshold_list[-clip_index:]:
                    file_name = threshold.name
                    shutil.move(
                        os.path.join(self.img_dir, file_name),
                        os.path.join(self.move_folder, file_name),
                    )
            else:
                for threshold in self.threshold_list[:-clip_index]:
                    file_name = threshold.name
                    shutil.move(
                        os.path.join(self.img_dir, file_name),
                        os.path.join(self.move_folder, file_name),
                    )
