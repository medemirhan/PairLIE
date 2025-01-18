import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from datetime import datetime
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.io as sio

def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    return gradient_h, gradient_w

def tv_loss(illumination):
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    loss_h = gradient_illu_h
    loss_w = gradient_illu_w
    loss = loss_h.mean() + loss_w.mean()
    return loss

def C_loss(R1, R2):
    loss = torch.nn.MSELoss()(R1, R2) 
    return loss

def R_loss(L1, R1, im1, X1):
    max_rgb1, _ = torch.max(im1, 1)
    max_rgb1 = max_rgb1.unsqueeze(1) 
    loss1 = torch.nn.MSELoss()(L1*R1, X1) + torch.nn.MSELoss()(R1, X1/L1.detach())
    loss2 = torch.nn.MSELoss()(L1, max_rgb1) + tv_loss(L1)
    return loss1 + loss2

def P_loss(im1, X1):
    loss = torch.nn.MSELoss()(im1, X1)
    return loss

def joint_RGB_horizontal(im1, im2):
    if im1.size==im2.size:
        w, h = im1.size
        result = Image.new('RGB',(w*2, h))
        result.paste(im1, box=(0,0))
        result.paste(im2, box=(w,0))      
    return result

def joint_L_horizontal(im1, im2):
    if im1.size==im2.size:
        w, h = im1.size
        result = Image.new('L',(w*2, h))
        result.paste(im1, box=(0,0))
        result.paste(im2, box=(w,0))   
    return result

class AvgMeter(object):
    # source: https://github.com/joeylitalien/noise2noise-pytorch
    """Computes and stores the average and current value.
    Useful for tracking averages such as elapsed times, minibatch losses, etc.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def clear_line():
    # source: https://github.com/joeylitalien/noise2noise-pytorch
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')

def show_on_report(batch_idx, num_batches, loss_total, loss_c, loss_r, loss_p, elapsed):
    # source: https://github.com/joeylitalien/noise2noise-pytorch
    """Formats training stats."""

    clear_line()
    dec = int(np.ceil(np.log10(num_batches)))
    print('Batch {:>{dec}d} / {:d} | TotalLoss: {:>1.5f} | C_Loss: {:>1.5f} | R_Loss: {:>1.5f} | P_Loss: {:>1.5f} | AvgTime/btch: {:d}ms'.format(batch_idx, num_batches, loss_total, loss_c, loss_r, loss_p, int(elapsed), dec=dec))

def time_elapsed_since(start):
    # source: https://github.com/joeylitalien/noise2noise-pytorch
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms

def progress_bar(batch_idx, num_batches, report_interval, lossTotal, loss1, loss2, loss3):
    # source: https://github.com/joeylitalien/noise2noise-pytorch
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 21 + dec
    progress = (batch_idx % report_interval) / report_interval
    fill = int(progress * bar_size) + 1
    print('\rBatch {:>{dec}d} / {:d} [{}{}] Loss: {:>1.5f} | Loss1: {:>1.5f} | Loss2: {:>1.5f} | Loss3: {:>1.5f}'.format(batch_idx, num_batches, '=' * fill + '>', ' ' * (bar_size - fill), lossTotal, loss1, loss2, loss3, dec=str(dec)), end='')

def show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr):
    # source: https://github.com/joeylitalien/noise2noise-pytorch
    """Formats validation error stats."""

    clear_line()
    print('Train time: {} | Valid time: {} | Valid loss: {:>1.5f} | Avg PSNR: {:.2f} dB'.format(epoch_time, valid_time, valid_loss, valid_psnr))

def plot_per_epoch(save_dir, title, measurements, y_label):
    # source: https://github.com/joeylitalien/noise2noise-pytorch
    """Plots stats (train/valid loss, avg PSNR, etc.)."""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(save_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()

def load_hsi_as_tensor(path, matContentHeader="data"):
    mat = sio.loadmat(path)
    mat = mat[matContentHeader]

    assert isinstance(mat, np.ndarray) and mat.dtype != np.uint8

    mat = torch.from_numpy(mat).to(dtype=torch.float32)
    mat = torch.permute(mat, (2, 0, 1))

    return mat

def plot_loss_curve(stats, save_path='loss_curves.png'):
    """Plot and save all training loss curves with epoch numbers"""
    
    epochs = range(1, len(stats['loss_total']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plot each loss in a separate subplot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, stats['loss_total'], 'k-', label='Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, stats['loss_c'], 'r-', label='c_loss')
    plt.title('Reflectance Consistency Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, stats['loss_r'], 'b-', label='r_loss')
    plt.title('Retinex Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, stats['loss_p'], 'g-', label='p_loss')
    plt.title('Projection Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()

class Struct:
    pass