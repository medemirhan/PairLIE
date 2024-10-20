import os
import numpy as np
from scipy.io import loadmat, savemat

def divideHsiOverlapping(input_dir, output_train_dir, output_test_dir, total_bands, output_band_num, overlap, reserved_for_test=None):
    
    # Ensure overlap is smaller than the output band count
    if overlap >= output_band_num:
        raise ValueError('Overlap must be smaller than the output band count')

    output_train_dir = f"{output_train_dir}_{output_band_num}_bands"
    output_test_dir = f"{output_test_dir}_{output_band_num}_bands"
    dirs = os.listdir(input_dir)

    dir_count = 1
    for d in dirs:
        cur_dir = os.path.join(input_dir, d)
        if os.path.isdir(cur_dir):
            file_list = [f for f in os.listdir(cur_dir) if f.endswith('.mat')]

            for cur_file in file_list:
                file_prefix, _ = os.path.splitext(cur_file)
                data_org = loadmat(os.path.join(cur_dir, cur_file))['data']
                start_band = 0
                
                while total_bands - start_band >= output_band_num * 2 - overlap:
                    for _ in range(2):
                        end_band = start_band + output_band_num
                        data = data_org[:, :, start_band:end_band]
                        generated_img_name = file_prefix

                        for m in range(start_band + 1, end_band + 1):
                            generated_img_name += f"_{m}"
                        generated_img_name += '.mat'

                        if cur_file in reserved_for_test:
                            test_dir_num = reserved_for_test.index(cur_file) + 1
                            save_path = os.path.join(output_test_dir, str(test_dir_num))
                        else:
                            save_path = os.path.join(output_train_dir, str(dir_count))

                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        savemat(os.path.join(save_path, generated_img_name), {'data': data})
                        start_band = end_band - overlap

                    if cur_file not in reserved_for_test:
                        dir_count += 1

def mergeOverlappingHsis(input_dir, input_file_prefix, total_bands, output_band_num, overlap):

    # Ensure overlap is smaller than the output band count
    if overlap >= output_band_num:
        raise ValueError('Overlap must be smaller than the output band count')
    
    start_band = 0
    data = None
    while total_bands - start_band >= output_band_num:
        end_band = start_band + output_band_num
        generated_img_name = input_file_prefix

        for m in range(start_band + 1, end_band + 1):
            generated_img_name += f"_{m}"
        generated_img_name += '.mat'
        
        if not os.path.exists(os.path.join(input_dir, generated_img_name)):
            raise FileNotFoundError(f"The file '{os.path.join(input_dir, generated_img_name)}' was not found.")
        
        data_org = loadmat(os.path.join(input_dir, generated_img_name))['data']

        if start_band == 0:
            data = data_org
        else:
            for j in range(overlap):
                temp = (data_org[:, :, j] + data[:, :, output_band_num - overlap + j]) / 2
                data[:, :, output_band_num - overlap + j] = temp
            data = np.concatenate((data, data_org[:, :, overlap:output_band_num]), axis=2)
        
        start_band = end_band - overlap

    # Save the merged data
    output_file = os.path.join(input_dir, input_file_prefix + '.mat')
    savemat(output_file, {'data': data})

if __name__ == '__main__':

    '''
    # Driver code for divideHsiOverlapping

    input_dir = 'hsi_dataset'
    output_train_dir = 'train_ll_overlap'
    output_test_dir = 'test_ll_overlap'
    total_bands = 64
    output_band_num = 8
    overlap = 3
    reserved_for_test = ['007_2_2021-01-20_024_renamed.mat', 'buildingblock_1ms_renamed.mat']

    divideHsiOverlapping(input_dir, output_train_dir, output_test_dir, total_bands, output_band_num, overlap, reserved_for_test)

    '''
    # Driver code for mergeOverlappingHsis

    input_dir = 'results/test_ll_overlap_8_bands_0_8_gelu_newLoss_norm/2/I'
    #input_file_prefix = '007_2_2021-01-20_024_renamed'
    input_file_prefix = 'buildingblock_1ms_renamed'
    total_bands = 63
    output_band_num = 8
    overlap = 3

    mergeOverlappingHsis(input_dir, input_file_prefix, total_bands, output_band_num, overlap)
    