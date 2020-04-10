import numpy as np
from torch.utils.data import Dataset


class DigitsDataset(Dataset):
    def __init__(self, data, transform, shuffle = True):
        self.data = data.sample(frac=1).reset_index(drop=True) if shuffle else data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        pixels = np.array(item[1:]).astype(np.uint8).reshape((28, 28))
        label = item[0]

        if self.transform is not None:
            pixels = self.transform(pixels)

        return pixels, label
