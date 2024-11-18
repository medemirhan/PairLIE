import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import argparse
#from thop import profile
from net.net import net
from data import get_eval_set, get_eval_set_hsi
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *
import eval
import hsiManipulations
import measure

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

    data_test = 'test_ll_skip_bands_outdoor'
    model = 'weights/train_20241104_003417_23db/epoch_180.pth'
    inp_channels = 3
    output_folder_prefix = 'results/test_ll_skip_bands_outdoor_'
    filename_trim = ['_renamed_', '_1ms_renamed_', '_renamed', '_1ms_renamed', '_']
    label_dir = 'label_ll'
    result_summary = "gridSearch.txt"

    luminance_factors = np.arange(0.1, 2.1, 0.1)

    for lf in luminance_factors:
        # eval.py
        output_folder = output_folder_prefix + "{:.1f}".format(lf)
        run_eval(data_test, model, output_folder, inp_channels, lf)
        
        # hsiManipulations.py
        results_folder = output_folder + '/I'
        combined_folder = output_folder + '/combined'
        hsiManipulations.concatenate_mat_files(results_folder, combined_folder, inp_channels, filename_trim)

        # measure.py
        measure_path = combined_folder + '/*.mat'
        avg_psnr, avg_ssim, avg_sam = measure.metrics_hsi(os.path.normpath(measure_path), os.path.normpath(label_dir))
        
        str_lf = "{:.1f}".format(lf)
        str_psnr = "{:.4f} dB".format(avg_psnr)
        str_ssim = "{:.4f}".format(avg_ssim)
        str_sam = "{:.4f}".format(avg_sam)

        line = f"luminance_factor={str_lf} , psnr={str_psnr} , mssim={str_ssim} , msam={str_sam}"

        with open(result_summary, "a") as file:
            file.write(line + "\n")