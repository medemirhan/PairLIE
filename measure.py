import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
import glob
import numpy as np
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, SpectralAngleMapper
import torch
import utils

def ssim(input, target, data_range=None):
    if data_range == None:
        ssim_torch = StructuralSimilarityIndexMeasure()
    else:
        ssim_torch = StructuralSimilarityIndexMeasure(data_range=data_range)

    return ssim_torch(input, target)

def psnr(input, target, data_range=None):
    if data_range == None:
        psnr_torch = PeakSignalNoiseRatio()
    else:
        psnr_torch = PeakSignalNoiseRatio(data_range=data_range)

    return psnr_torch(input, target)

def sam(input, target, reduction='elementwise_mean'):
    sam_torch = SpectralAngleMapper(reduction=reduction)
    return sam_torch(input, target)

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
        score_ssim = ssim(im1, im2)
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n

    return avg_psnr, avg_ssim

def metrics_hsi(im_dir, label_dir):
    avg_psnr = 0
    avg_ssim = 0
    avg_sam = 0
    n = 0
    for item in sorted(glob.glob(im_dir)):
        n += 1
        im1 = utils.load_hsi_as_tensor(item)
        name = item.split('\\')[-1]
        im2 = utils.load_hsi_as_tensor(os.path.join(label_dir, name))

        im1 = im1.unsqueeze(0)
        im2 = im2.unsqueeze(0)

        score_psnr = psnr(im1, im2, data_range=torch.max(torch.max(im1),torch.max(im2))) # data range onemli. incele!
        score_ssim = ssim(im1, im2, data_range=torch.max(torch.max(im1),torch.max(im2))) # data range onemli. incele!
        score_sam = sam(im1, im2, reduction='elementwise_mean') # reduction onemli. incele!
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_sam += score_sam

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_sam = avg_sam / n

    return avg_psnr, avg_ssim, avg_sam

if __name__ == '__main__':

    #im_dir = 'PairLIE-our-results/LOL-test/I/*.png'
    #label_dir = 'PairLIE-testing-dataset/LOL-test/reference'

    im_dir = 'results/test_ll_overlap_10_bands/1/I/*.mat'
    label_dir = 'test_ll_overlap_10_bands/1'

    avg_psnr, avg_ssim, avg_sam = metrics_hsi(os.path.normpath(im_dir), os.path.normpath(label_dir))
    print("===> Avg.PSNR : {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM : {:.4f} ".format(avg_ssim))
    print("===> Avg.SAM  : {:.4f} ".format(avg_sam))
