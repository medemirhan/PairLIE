import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#import torch.backends.cudnn as cudnn
import torch
torch.cuda.is_available()
from torch.utils.data import DataLoader
from net.net import net
from data import get_training_set, get_training_set_hsi, get_eval_set, get_eval_set_hsi
from utils import *
import main
import eval
import hsiManipulations
import measure
import mlflow

REPORT_INTERVAL = 10

def run_train(nEpochs, lr, data_train, inp_channels, cLossCoeff, rLossCoeff, pLossCoeff):
    params = main.parse_args()

    params.batchSize = 1
    params.nEpochs = nEpochs
    params.snapshots = 5
    params.start_iter = 1
    params.lr = lr
    params.gpu_mode = True
    params.threads = 0
    params.decay = 50
    params.gamma = 0.5
    params.seed = 42
    #params.data_train = 'PairLIE-training-dataset'
    params.data_train = data_train
    params.inp_channels = inp_channels
    params.num_3d_filters = 16
    params.num_conv_filters = 10
    params.rgb_range = 1
    params.save_folder = 'weights'
    params.output_folder = 'results'

    params.cLossCoeff = cLossCoeff
    params.rLossCoeff = rLossCoeff
    params.pLossCoeff = pLossCoeff

    main.seed_torch(params.seed)
    #cudnn.benchmark = True
    cuda = params.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    print('===> Loading datasets')
    #train_set = get_training_set(params.data_train)
    train_set = get_training_set_hsi(params.data_train)
    training_data_loader = DataLoader(dataset=train_set, num_workers=params.threads, batch_size=params.batchSize, shuffle=True)

    model_timestamp, model_dir, model_full_path = main.train(params, training_data_loader)

    return model_timestamp, model_dir, model_full_path

def run_eval(data_test, model, output_folder, inp_channels, luminance_factor):
    params = eval.parse_args()

    params.testBatchSize = 1
    params.gpu_mode = False
    params.threads = 0
    params.rgb_range = 1
    params.data_test = data_test
    params.model = model
    params.output_folder = output_folder
    params.inp_channels = inp_channels
    params.luminance_factor = luminance_factor
    params.num_3d_filters = 16
    params.num_conv_filters = 10

    print('===> Loading datasets')
    test_set = get_eval_set_hsi(params.data_test)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=params.threads, batch_size=params.testBatchSize, shuffle=False)

    print('===> Building model')
    if params.gpu_mode:
        model = net(params.inp_channels).cuda()
    else:
        model = net(params.inp_channels)
    model.load_state_dict(torch.load(params.model, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')

    eval.eval(params, model, testing_data_loader)

if __name__ == '__main__':
    
    # don't change
    filename_trim = ['_renamed_', '_1ms_renamed_', '_renamed', '_1ms_renamed', '_']

    # change
    nEpochs = 100
    lr = 1e-4
    data_train = 'train_ll_skip_bands_outdoor'
    data_test = 'test_ll_skip_bands_outdoor'
    test_label_dir = 'label_ll'
    inp_channels = 3

    cLossCoeff = [0.5]
    rLossCoeff = [75.0]
    pLossCoeff = [5.0]
    luminance_factors = [0.9]

    timestamp = f'{datetime.now():{""}%H%M%S}'
    exp_name = 'test_ll_skip_bands_outdoor'
    exp_purpose = exp_name + '_' + timestamp

    mlflow.set_experiment("/" + exp_purpose)
    for c in cLossCoeff:
        for r in rLossCoeff:
            for p in pLossCoeff:
                model_timestamp, model_dir, model_full_path = run_train(nEpochs, lr, data_train, inp_channels, c, r, p)

                model_path = model_full_path
                model_tag = model_timestamp

                exp_summary = exp_name + '_model_' + model_tag
                output_dir = 'D:/results/' + exp_summary

                for lf in luminance_factors:
                    with mlflow.start_run(nested=False):
                        # eval.py
                        output_folder = output_dir + "/lf_{:.1f}".format(lf)
                        run_eval(data_test, model_path, output_folder, inp_channels, lf)
                        
                        # hsiManipulations.py
                        results_folder = output_folder + '/I'
                        combined_folder = output_folder + '/combined'
                        hsiManipulations.concatenate_mat_files(results_folder, combined_folder, inp_channels, filename_trim)

                        # measure.py
                        measure_path = combined_folder + '/*.mat'
                        avg_psnr, avg_ssim, avg_sam = measure.metrics_hsi(os.path.normpath(measure_path), os.path.normpath(test_label_dir))

                        str_lf = "{:.1f}".format(lf)
                        str_psnr = "{:.4f} dB".format(avg_psnr)
                        str_ssim = "{:.4f}".format(avg_ssim)
                        str_sam = "{:.4f}".format(avg_sam)

                        mlflow.log_param("model_tag", model_tag)
                        mlflow.log_param("cLossCoeff", c)
                        mlflow.log_param("rLossCoeff", r)
                        mlflow.log_param("pLossCoeff", p)
                        mlflow.log_param("lf", str_lf)
                        mlflow.log_param("avg_psnr_dB", avg_psnr.item())
                        mlflow.log_param("avg_ssim", avg_ssim.item())
                        mlflow.log_param("avg_sam", avg_sam.item())
                        mlflow.log_param("data_train", data_train)
                        mlflow.log_param("data_test", data_test)
                        mlflow.log_param("nEpochs", nEpochs)
                        mlflow.log_param("lr", lr)
                        mlflow.log_param("inp_channels", inp_channels)
