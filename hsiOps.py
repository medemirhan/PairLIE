import os
import numpy as np
from scipy.io import loadmat, savemat
import re

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

def generate_banded_mat_files_for_all(directory, save_dir, num_bands, reserved_for_test, key='data'):
    """
    Processes all `.mat` files in a specified directory,
    creating new .mat files by selecting a user-defined number of bands from each original file in pairs.
    The function operates as follows:

    Inputs:
    - `directory`: Path to the directory containing the original .mat files to be processed.
    - `save_dir`: Path where newly created folders containing the output .mat files will be saved.
    - `num_bands`: Number of bands to include in each new .mat file.
    #
    Outputs:
    - The function saves pairs of new .mat files into incrementally numbered folders within the `save_dir` directory.
    - Each new file contains a subset of bands from the original file, with file names indicating the selected band indices.
    - Folder numbering continues across all files in the directory, avoiding resets.
    #
    Example Usage:
    Suppose we have three `.mat` files (`file1.mat`, `file2.mat`, and `file3.mat`) in a directory, 
    and we want to create files with 3-band selections saved in `/output_folder`.
    #
    Call:
    create_banded_mat_files_for_all('/path/to/mat_files', '/output_folder', 3)
    #
    Example Outputs:
    1. For `file1.mat`, the function may create folders `/output_folder/1`, `/output_folder/2`, etc., containing:
        - `file1_1_3_5.mat`, `file1_2_4_6.mat` in folder `1`
        - `file1_7_9_11.mat`, `file1_8_10_12.mat` in folder `2`
    2. For `file2.mat`, the folder numbering continues from where `file1.mat` left off, creating:
        - `file2_1_3_5.mat`, `file2_2_4_6.mat` in `/output_folder/3`
        - `file2_7_9_11.mat`, `file2_8_10_12.mat` in `/output_folder/4`
    3. This pattern continues for each `.mat` file, ensuring unique folder numbering across all files.
    """
    # Get all .mat files in the specified directory
    mat_files = [f for f in os.listdir(directory) if f.endswith('.mat')]
    
    if not mat_files:
        print("No .mat files found in the specified directory.")
        return
    
    folder_index = 1  # Initialize folder numbering

    # Process each .mat file in the directory
    for mat_file in mat_files:
        if mat_file in reserved_for_test:
            continue
        
        mat_path = os.path.join(directory, mat_file)
        filename = os.path.splitext(mat_file)[0]  # Get filename without extension
        
        # Load the MAT file
        mat_data = loadmat(mat_path)
        if key not in mat_data:
            print(f"The '{key}' key is not found in {mat_file}. Skipping this file.")
            continue
        data = mat_data[key]
        
        # Check the dimensions of the data
        if len(data.shape) != 3:
            print(f"The data in {mat_file} is not in nxmxc format. Skipping this file.")
            continue
        n, m, c = data.shape

        # Create new files in pairs and save them in numbered folders
        for i in range(0, c, num_bands * 2):
            # Create two sets of bands with a stride of 2
            set1 = [data[:, :, i + j] for j in range(0, num_bands * 2, 2) if (i + j) < c]
            set2 = [data[:, :, i + j + 1] for j in range(0, num_bands * 2, 2) if (i + j + 1) < c]

            # Skip if there are not enough bands to create the required number of bands
            if len(set1) != num_bands or len(set2) != num_bands:
                break

            # Stack the selected bands along the third dimension to form new matrices
            set1 = np.stack(set1, axis=-1)
            set2 = np.stack(set2, axis=-1)

            # Create a numbered folder for this pair
            folder_path = os.path.join(save_dir, str(folder_index))
            os.makedirs(folder_path, exist_ok=True)

            # Generate filenames based on selected bands
            set1_bands = "_".join(str(i + 1 + j) for j in range(0, num_bands * 2, 2))
            set2_bands = "_".join(str(i + 2 + j) for j in range(0, num_bands * 2, 2))

            set1_filename = f"{filename}_{set1_bands}.mat"
            set2_filename = f"{filename}_{set2_bands}.mat"

            # Save the new .mat files in the specified folder
            savemat(os.path.join(folder_path, set1_filename), {'data': set1})
            savemat(os.path.join(folder_path, set2_filename), {'data': set2})

            print(f"Saved {set1_filename} and {set2_filename} to {folder_path}")
            folder_index += 1  # Increment the folder index for the next pair

def get_common_prefix(directory):
    # List to store the filenames
    filenames = []

    # Iterate over each file in the specified directory and collect relevant filenames
    for filename in os.listdir(directory):
        if filename.endswith('.mat'):
            filenames.append(filename)

    # Function to find the longest common prefix
    def longest_common_prefix(file_list):
        if not file_list:
            return ""

        # Start with the first file's name as the initial prefix
        prefix = file_list[0]

        for filename in file_list[1:]:
            while not filename.startswith(prefix):
                # Reduce the prefix until it matches
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        
        return prefix

    # Find the longest common prefix among the filenames
    common_prefix = longest_common_prefix(filenames)

    return common_prefix

def count_files_with_extension(directory, extension):
    # Normalize the extension (remove the leading dot if present)
    if extension.startswith('.'):
        extension = extension[1:]
    
    count = 0
    # Iterate through files in the given directory
    for filename in os.listdir(directory):
        # Check if the file ends with the specified extension
        if filename.endswith(f'.{extension}'):
            count += 1
            
    return count

def concatenate_mat_files(input_directory, output_directory, c, filename_trim):
    """
    Concatenate specific slices from .mat files in the input directory based on their filenames
    and save the result to a .mat file in the output directory.
    
    Each .mat file contains a "data" matrix with shape (n, m, k), where `k` specifies how many
    bands from this file should be mapped to specific positions in the final matrix. The filename
    indicates the target bands for each slice along the 3rd axis, with 1-based indices.

    Example file naming:
    - "filename_13_15_17.mat" contributes its slices to the 12th, 14th, and 16th bands 
      of the output matrix (after adjusting for 0-based indexing).

    Parameters:
    - input_directory (str): Directory containing .mat files.
    - output_directory (str): Directory to save the concatenated output file.
    - c (int): The size of the 3rd axis (depth) in the concatenated output matrix.
    
    Example usage:
    >>> concatenate_mat_files("path/to/input_directory", "path/to/output_directory", 20)
    
    The resulting .mat file will be saved as "concatenated_data.mat" in the output directory.
    """

    prefix = get_common_prefix(input_directory)
    for trim in filename_trim:
        prefix = prefix.strip(trim)

    num_files = count_files_with_extension(input_directory, 'mat')

    # Initialize an empty list with None for the slices along the 3rd axis
    n, m = None, None  # Dimensions of the matrix (n, m)
    concatenated_data = None

    # Regex to extract exactly c indices from the filename
    regex_string = '_(\\d+)'
    for i in range(c-1):
        regex_string += '_([\\d]+)'
    regex_string += '\\.mat$'

    pattern = re.compile(regex_string)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.mat'):
            # Match the filename to find specified indices
            match = pattern.search(filename)
            if match:
                # Parse indices from filename and adjust for 0-based indexing
                indices = []
                for i in range(c):
                    indices.append(int(match.group(i+1)) - 1)

                # Load the .mat file and get the "data" matrix
                file_path = os.path.join(input_directory, filename)
                mat_data = loadmat(file_path)
                
                # Ensure "data" exists in the file
                if "data" not in mat_data:
                    raise ValueError(f"'data' key not found in file {filename}")
                
                data = mat_data["data"]
                k = data.shape[2]  # Number of bands in this file

                # Get n and m from the first file, and validate against subsequent files
                if n is None and m is None:
                    n, m = data.shape[:2]
                elif data.shape[:2] != (n, m):
                    raise ValueError(f"File '{filename}' has incompatible dimensions {data.shape}.")

                if concatenated_data is None:
                    concatenated_data = np.empty((n, m, c*num_files))

                # Ensure the number of indices matches the 3rd dimension of data
                if len(indices) != k:
                    raise ValueError(f"File '{filename}' has {k} bands, but filename specifies {len(indices)} bands.")
                
                # Assign data slices to the specified positions in `slices`
                for i, idx in enumerate(indices):
                    concatenated_data[:, :, idx] = data[:, :, i]

    # Check if all positions are filled
    if None in concatenated_data:
        raise ValueError(f"Some bands are missing.")

    # Save the concatenated matrix as a .mat file in the output directory
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, prefix + ".mat")
    savemat(output_path, {"data": concatenated_data})
    print(f"Concatenated .mat file saved to {output_path}")

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
    '''# Driver code for mergeOverlappingHsis

    input_dir = 'results/test_ll_overlap_8_bands_0_8_gelu_newLoss_norm/2/I'
    #input_file_prefix = '007_2_2021-01-20_024_renamed'
    input_file_prefix = 'buildingblock_1ms_renamed'
    total_bands = 63
    output_band_num = 8
    overlap = 3

    mergeOverlappingHsis(input_dir, input_file_prefix, total_bands, output_band_num, overlap)'''

    '''# Driver code for generate_banded_mat_files_for_all
    directory = 'hsi_dataset/outdoor'
    save_dir = 'train_ll_skip_bands_outdoor_6_bands'
    num_bands = 6
    reserved_for_test = ['007_2_2021-01-20_024_renamed.mat', 'buildingblock_1ms_renamed.mat']
    generate_banded_mat_files_for_all(directory, save_dir, num_bands, reserved_for_test)'''

    # Driver code for concatenate_mat_files
    input_directory = 'results/test_ll_skip_bands_outdoor_6_bands/I'
    output_directory = 'results/test_ll_skip_bands_outdoor_6_bands/combined'
    c = 6
    filename_trim = ['_renamed_', '_1ms_renamed_', '_renamed', '_1ms_renamed', '_']
    concatenate_mat_files(input_directory, output_directory, c, filename_trim)

    