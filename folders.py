import PIL.Image
import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
import torchvision
from torchvision import transforms

class cbs(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        sample = []
        rp = '/media/venn/ourdata/hj/data'
        with open('/media/venn/ourdata/hj/data/nr.txt', 'r') as listFile:
            for line in listFile:
                scn_idx, dis, score = line.split()
                scn_idx = int(scn_idx)
                score = float(score)
                if scn_idx in index:
                    sample.append((os.path.join(rp, dis), score))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
