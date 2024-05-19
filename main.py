import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
import random
import shutil
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from net.net import net
from data import get_training_set, get_eval_set
from utils import *
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter

REPORT_INTERVAL = 3

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PairLIE')
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--snapshots', type=int, default=20, help='Snapshots')
    parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--decay', type=int, default='100', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
    parser.add_argument('--seed', type=int, default=123456789, help='random seed to use. Default=123')
    parser.add_argument('--data_train', type=str, default='../dataset/PairLIE-training-dataset/')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    parser.add_argument('--output_folder', default='results/', help='Location to save checkpoint models')
    return parser.parse_args()

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_on_epoch(epoch, model, writer, training_data_loader, optimizer, stats, save_dir):
    train_loss1_meter = AvgMeter()
    train_loss2_meter = AvgMeter()
    train_loss3_meter = AvgMeter()
    train_lossTotal_meter = AvgMeter()
    loss1_meter = AvgMeter()
    loss2_meter = AvgMeter()
    loss3_meter = AvgMeter()
    lossTotal_meter = AvgMeter()
    time_meter = AvgMeter()

    num_batches = len(training_data_loader)

    model.train()
    loss_print = 0
    epoch_start = datetime.now()
    for iteration, batch in enumerate(training_data_loader, 1):
        batch_start = datetime.now()
        progress_bar(iteration, num_batches, REPORT_INTERVAL, lossTotal_meter.val, loss1_meter.val, loss2_meter.val, loss3_meter.val)

        im1, im2, file1, file2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        L1, R1, X1 = model(im1)
        L2, R2, X2 = model(im2)   
        loss1 = C_loss(R1, R2)
        loss2 = R_loss(L1, R1, im1, X1)
        loss3 = P_loss(im1, X1)
        loss =  loss1 * 1 + loss2 * 1 + loss3 * 500

        loss1_meter.update(loss1.item())
        loss2_meter.update(loss2.item())
        loss3_meter.update(loss3.item())
        lossTotal_meter.update(loss.item())

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("C_loss/train", loss1, epoch)
        writer.add_scalar("R_loss/train", loss2, epoch)
        writer.add_scalar("P_loss/train", loss3, epoch)
        writer.add_scalar("Lr/train", optimizer.param_groups[0]['lr'], epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_print = loss_print + loss.item()
        if iteration % REPORT_INTERVAL == 0:
            '''print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                iteration, num_batches, loss_print, optimizer.param_groups[0]['lr']))'''
            loss_print = 0

        # Report/update statistics
        time_meter.update(time_elapsed_since(batch_start)[1])
        if iteration % REPORT_INTERVAL == 0 and iteration:
            show_on_report(iteration, num_batches, lossTotal_meter.avg, loss1_meter.avg, loss2_meter.avg, loss3_meter.avg,time_meter.avg)
            train_lossTotal_meter.update(lossTotal_meter.avg)
            train_loss1_meter.update(loss1_meter.avg)
            train_loss2_meter.update(loss2_meter.avg)
            train_loss3_meter.update(loss3_meter.avg)
            lossTotal_meter.reset()
            loss1_meter.reset()
            loss2_meter.reset()
            loss3_meter.reset()
            time_meter.reset()

    # Epoch end, save and reset tracker
    on_epoch_end(stats, train_loss1_meter.avg, train_loss2_meter.avg, train_loss3_meter.avg, train_lossTotal_meter.avg, save_dir)
    train_lossTotal_meter.reset()
    train_loss1_meter.reset()
    train_loss2_meter.reset()
    train_loss3_meter.reset()

def on_epoch_end(stats, loss1, loss2, loss3, total_loss, save_dir):
    """Tracks and saves starts after each epoch."""

    # Save checkpoint
    stats['loss1'].append(loss1)
    stats['loss2'].append(loss2)
    stats['loss3'].append(loss3)
    stats['total_loss'].append(total_loss)
    
    plot_per_epoch(save_dir, 'c_loss', stats['loss1'], 'C Loss')
    plot_per_epoch(save_dir, 'r_loss', stats['loss2'], 'R Loss')
    plot_per_epoch(save_dir, 'p_loss', stats['loss3'], 'P Loss')
    plot_per_epoch(save_dir, 'training_loss', stats['total_loss'], 'Training Loss')

def checkpoint(epoch, model_state_dict, dir):
    os.makedirs(dir, exist_ok=True)
    model_out_path = os.path.join(dir, "epoch_{}.pth".format(epoch))
    torch.save(model_state_dict, model_out_path)
    print("\nCheckpoint saved to {}".format(model_out_path))

def train(params, training_data_loader):
    print('===> Building model ')
    model = net().cuda()
    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-8)

    model_timestamp = f'{datetime.now():{""}%Y%m%d_%H%M%S}'
    save_dir = os.path.normpath(os.path.join(params.save_folder, 'train_' + model_timestamp))

    milestones = []
    for i in range(1, params.nEpochs+1):
        if i % params.decay == 0:
            milestones.append(i)

    scheduler = lrs.MultiStepLR(optimizer, milestones, params.gamma)

    # Dictionaries of tracked stats
    stats = {'total_loss': [],
             'loss1': [],
             'loss2': [],
             'loss3': []}

    writer = SummaryWriter()
    train_start = datetime.now()
    for epoch in range(params.start_iter, params.nEpochs + 1):
        print('\nEPOCH {:d} / {:d}'.format(epoch, params.nEpochs))
        train_on_epoch(epoch, model, writer, training_data_loader, optimizer, stats, save_dir)
        scheduler.step()
        if epoch % params.snapshots == 0:
            checkpoint(epoch, model.state_dict(), save_dir)
    
    train_elapsed = time_elapsed_since(train_start)[0]

    writer.flush()
    writer.close()

    print('\nTraining done! Total elapsed time: {}\n'.format(train_elapsed))

if __name__ == '__main__':
    params = parse_args()

    params.batchSize = 1
    params.nEpochs = 50
    params.snapshots = 5
    params.start_iter = 1
    params.lr = 1e-4
    params.gpu_mode = True
    params.threads = 0
    params.decay = 100
    params.gamma = 0.5
    params.seed = 42
    params.data_train = 'PairLIE-training-dataset'
    params.rgb_range = 1
    params.save_folder = 'weights'
    params.output_folder = 'results'

    seed_torch(params.seed)
    cudnn.benchmark = True
    cuda = params.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    print('===> Loading datasets')
    train_set = get_training_set(params.data_train)
    training_data_loader = DataLoader(dataset=train_set, num_workers=params.threads, batch_size=params.batchSize, shuffle=True)

    train(params, training_data_loader)
