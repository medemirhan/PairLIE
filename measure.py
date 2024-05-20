import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
import glob
import numpy as np
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch

def ssim(input, target, data_range=None):

    if torch.is_tensor(input):
        data_in = input
    else:
        data_in = torch.tensor(input)

    if torch.is_tensor(target):
        data_out = target
    else:
        data_out = torch.tensor(target)

    if data_range == None:
        psnr_torch = StructuralSimilarityIndexMeasure()
    else:
        psnr_torch = StructuralSimilarityIndexMeasure(data_range=data_range)

    return psnr_torch(data_in, data_out)

def psnr(input, target, data_range=None):

    if torch.is_tensor(input):
        data_in = input
    else:
        data_in = torch.tensor(input)

    if torch.is_tensor(target):
        data_out = target
    else:
        data_out = torch.tensor(target)

    if data_range == None:
        psnr_torch = PeakSignalNoiseRatio()
    else:
        psnr_torch = PeakSignalNoiseRatio(data_range=data_range)

    return psnr_torch(data_in, data_out)

def metrics(im_dir, label_dir):
    avg_psnr = 0
    avg_ssim = 0
    n = 0

    for item in sorted(glob.glob(im_dir)):
        n += 1
        im1 = Image.open(item).convert('RGB') 

        name = item.split('\\')[-1]

        im2 = Image.open(os.path.join(label_dir, name)).convert('RGB')
        (h, w) = im2.size
        im1 = im1.resize((h, w))  
        im1 = np.array(im1, dtype=np.float32)
        im2 = np.array(im2, dtype=np.float32)

        im1 = torch.tensor(im1).unsqueeze(0).permute(0, 3, 1, 2)
        im2 = torch.tensor(im2).unsqueeze(0).permute(0, 3, 1, 2)

        score_psnr = psnr(im1, im2)
        ss = StructuralSimilarityIndexMeasure()
        score_ssim = ss(im1, im2)
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n

    return avg_psnr, avg_ssim

if __name__ == '__main__':

    im_dir = 'PairLIE-our-results/LOL-test/I/*.png'
    label_dir = 'PairLIE-testing-dataset/LOL-test/reference'

    #im_dir = 'results/sonuc/*.mat'
    #label_dir = 'test_reference'

    avg_psnr, avg_ssim = metrics(os.path.normpath(im_dir), os.path.normpath(label_dir))
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
