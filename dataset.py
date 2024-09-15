
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torchvision.transforms as transforms
import utils


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg", ".mat"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        index = index
        data_filenames = [join(join(self.data_dir, str(index+1)), x) for x in listdir(join(self.data_dir, str(index+1))) if is_image_file(x)]
        num = len(data_filenames)
        index1 = random.randint(1,num)
        index2 = random.randint(1,num)
        while abs(index1 - index2) == 0:
            index2 = random.randint(1,num)

<<<<<<< Updated upstream
<<<<<<< Updated upstream
        im1 = load_img(data_filenames[index1-1])
        im2 = load_img(data_filenames[index2-1])
=======
=======
>>>>>>> Stashed changes
        im1 = load_img(data_filenames[index1 - 1])
        im2 = load_img(data_filenames[index2 - 1])
        #im1 = utils.load_hsi_as_tensor(data_filenames[index1 - 1])
        #im2 = utils.load_hsi_as_tensor(data_filenames[index2 - 1])

        #im1 = im1 / torch.max(im1)
        #im2 = im2 / torch.max(im2)
>>>>>>> Stashed changes

        _, file1 = os.path.split(data_filenames[index1-1])
        _, file2 = os.path.split(data_filenames[index2-1])

        seed = np.random.randint(42) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)        
        return im1, im2, file1, file2

    def __len__(self):
        return len(os.listdir(self.data_dir)) # for custom datasets, please check the dataset size and modify this number


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes
        #input = utils.load_hsi_as_tensor(self.data_filenames[index])
        #input = input / torch.max(input)
>>>>>>> Stashed changes
        _, file = os.path.split(self.data_filenames[index])

        if self.transform:
            input = self.transform(input)
        return input, file

    def __len__(self):
        return len(self.data_filenames)

