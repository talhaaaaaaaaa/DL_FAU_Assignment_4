from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision.transforms as tv
import pandas as pd

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data: pd.DataFrame, mode: str): # mode ['val', 'train']
        self.data = data
        self.mode = mode

        self.mean = train_mean[0]
        self.std = train_std[0]

        # Define transformations
        if self.mode == 'train':
            self.transform = tv.Compose([
                tv.ToPILImage(),
                tv.ToTensor(),
                tv.Normalize(mean=[self.mean], std=[self.std])
            ])
        else:  # Validation transformations
            self.transform = tv.Compose([
                tv.ToPILImage(),
                tv.ToTensor(),
                tv.Normalize(mean=[self.mean], std=[self.std])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        image_path = row['filename']
        labels = row[['crack', 'inactive']].values.astype(np.float32)

        image = imread(image_path)
        image = gray2rgb(image)

        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        labels = torch.tensor(labels, dtype=torch.long)

        return image, labels


data = pd.read_csv(r'C:\Users\User\OneDrive\Dokumente\Wintersemester_24_25\DeepLearning\exercise4_material\src_to_implement\data.csv')
train_dataset = ChallengeDataset(data=data, mode='train')
val_dataset = ChallengeDataset(data=data, mode='val')

