from torchvision.transforms import Compose, ToTensor, RandomCrop
from dataset import DatasetFromFolderEval, DatasetFromFolder
import numpy as np

def transform1():
    return Compose([
        RandomCrop((128, 128)),
        ToTensor(),
    ])

def transform2():
    return Compose([
        ToTensor(),
    ])

def transform3():
    return Compose([
        RandomCrop((128, 128)),
    ])

def transform4():
    return Compose([
    ])

def get_training_set(data_dir):
    return DatasetFromFolder(data_dir, transform=transform1())

def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())

def get_training_set_hsi(data_dir):
    return DatasetFromFolder(data_dir, transform=transform3())

def get_eval_set_hsi(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform4())