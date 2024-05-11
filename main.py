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

def train_on_epoch(epoch, model, training_data_loader, optimizer):
    model.train()
    loss_print = 0
    for iteration, batch in enumerate(training_data_loader, 1):

        im1, im2, file1, file2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        L1, R1, X1 = model(im1)
        L2, R2, X2 = model(im2)   
        loss1 = C_loss(R1, R2)
        loss2 = R_loss(L1, R1, im1, X1)
        loss3 = P_loss(im1, X1)
        loss =  loss1 * 1 + loss2 * 1 + loss3 * 500

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_print = loss_print + loss.item()
        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                iteration, len(training_data_loader), loss_print, optimizer.param_groups[0]['lr']))
            loss_print = 0

def checkpoint(epoch, model_state_dict, save_folder, timestamp):
    dir = os.path.normpath(os.path.join(save_folder, 'train_' + timestamp))
    os.makedirs(dir, exist_ok=True)
    model_out_path = os.path.join(dir, "epoch_{}.pth".format(epoch))
    torch.save(model_state_dict, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def train(params, training_data_loader):
    print('===> Building model ')
    model = net().cuda()
    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-8)

    model_timestamp = f'{datetime.now():{""}%Y%m%d_%H%M%S}'

    milestones = []
    for i in range(1, params.nEpochs+1):
        if i % params.decay == 0:
            milestones.append(i)

    scheduler = lrs.MultiStepLR(optimizer, milestones, params.gamma)

    score_best = 0
    # shutil.rmtree(params.save_folder)
    # os.mkdir(params.save_folder)
    for epoch in range(params.start_iter, params.nEpochs + 1):
        train_on_epoch(epoch, model, training_data_loader, optimizer)
        scheduler.step()
        if epoch % params.snapshots == 0:
            checkpoint(epoch, model.state_dict(), params.save_folder, model_timestamp)          

if __name__ == '__main__':
    params = parse_args()

    params.batchSize = 1
    params.nEpochs = 100
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