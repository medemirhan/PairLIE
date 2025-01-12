import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
import glob
import numpy as np
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, SpectralAngleMapper
import torch
import utils
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def ssim(input, target, data_range=None):
    if data_range == None:
        ssim_torch = StructuralSimilarityIndexMeasure()
    else:
        ssim_torch = StructuralSimilarityIndexMeasure(data_range=data_range)

    return ssim_torch(input.unsqueeze(0), target.unsqueeze(0))

def ssim_sk(input, target, data_range=None):
    if data_range == None:
        return structural_similarity(input, target)
    else:
        return structural_similarity(input, target, data_range=data_range[1] - data_range[0])

def psnr(input, target, data_range=None):
    if data_range == None:
        psnr_torch = PeakSignalNoiseRatio()
    else:
        psnr_torch = PeakSignalNoiseRatio(data_range=data_range)

    return psnr_torch(input, target)

def psnr_sk(input, target, data_range=None):
    if data_range == None:
        return peak_signal_noise_ratio(input, target)
    else:
        return peak_signal_noise_ratio(input, target, data_range=data_range[1] - data_range[0])

def sam(input, target, reduction='elementwise_mean'):
    sam_torch = SpectralAngleMapper(reduction=reduction)
    return sam_torch(input.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0))

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

def metrics_hsi(im_dir, label_dir, data_min=None, data_max=None, matKeyPrediction='data', matKeyGt='data'):    
    avg_psnr = 0
    avg_ssim = 0
    avg_sam = 0
    n = 0
    for item in sorted(glob.glob(im_dir)):
        n += 1
        im1 = utils.load_hsi_as_tensor(item, matContentHeader=matKeyPrediction)
        name = item.split('\\')[-1]
        im2 = utils.load_hsi_as_tensor(os.path.join(label_dir, name), matContentHeader=matKeyGt)
        
        if im1.shape[0] < im2.shape[0]:
            im2 = im2[:im1.shape[0], :, :]

        '''im1 = im1.unsqueeze(0)
        im2 = im2.unsqueeze(0)'''

        data_range = None
        if data_min != None and data_max != None:
            data_range = (data_min, data_max)
        elif data_max != None:
            data_range = data_max

        '''score_psnr = psnr_sk(im1.squeeze(0).detach().cpu().numpy(), im2.squeeze(0).detach().cpu().numpy(), data_range=data_range) # data range onemli. incele!
        score_ssim = ssim_sk(im1.squeeze(0).detach().cpu().numpy(), im2.squeeze(0).detach().cpu().numpy(), data_range=data_range) # data range onemli. incele!
        score_sam = sam(im1, im2, reduction='elementwise_mean') # reduction onemli. incele!'''

        score_psnr = psnr(im1, im2, data_range=data_range) # data range onemli. incele!
        score_ssim = ssim(im1, im2, data_range=data_range) # data range onemli. incele!
        score_sam = sam(im1, im2, reduction='elementwise_mean') # reduction onemli. incele!
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_sam += score_sam

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_sam = avg_sam / n

    return avg_psnr, avg_ssim, avg_sam

if __name__ == '__main__':

    globalMin = 0.0708354
    globalMax = 1.7410845
    lowLightMin = 0.0708354
    lowLightMax = 0.2173913
    normalLightMin = 0.0708354
    normalLightMax = 1.7410845

    globalMin = 0.
    globalMax = 0.005019044472441
    
    '''im_dir = '../Self-supervised-Image-Enhancement-Network-Training-With-Low-Light-Images-Only/data/test_results/non_scaled/renamed/*.mat'
    label_dir = 'data/label_ll'''

    im_dir = 'D:/sslie/test_results_2nd/non_scaled/renamed/*.mat'
    label_dir = 'data/CZ_hsdb/lowered_1.9/gt'

    avg_psnr, avg_ssim, avg_sam = metrics_hsi(os.path.normpath(im_dir), os.path.normpath(label_dir), data_max=globalMax, matKeyPrediction='ref', matKeyGt='ref')
    #avg_psnr2, avg_ssim2, avg_sam2 = metrics_hsi(os.path.normpath(im_dir), os.path.normpath(label_dir), matKeyPrediction='ref', matKeyGt='data')

    #avg_psnr, avg_ssim, avg_sam = metrics_hsi(os.path.normpath(im_dir), os.path.normpath(label_dir), matKeyPrediction='ref', matKeyGt='data')
    print("\n===> Avg.PSNR : {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM : {:.4f} ".format(avg_ssim))
    print("===> Avg.SAM  : {:.4f} ".format(avg_sam))
