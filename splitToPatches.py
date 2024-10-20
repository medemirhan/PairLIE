import os
import numpy as np
import scipy.io as sio

def split_matrix_to_patches(input_file_path, h_input, w_input, h_out, w_out, crop_direction, output_folder):
    # Load the .mat file and determine the variable name automatically
    mat_contents = sio.loadmat(input_file_path)
    var_name = [key for key in mat_contents.keys() if not key.startswith('__')]
    if len(var_name) != 1:
        raise ValueError(f"Expected exactly one variable in the .mat file, but found {len(var_name)} variables.")
    var_name = var_name[0]
    
    # Extract the matrix using the identified variable name
    matrix = mat_contents[var_name]
    
    # Check if input matrix matches the given dimensions (h_input, w_input)
    h, w, c = matrix.shape
    if h != h_input or w != w_input:
        raise ValueError(f"The matrix dimensions {h}x{w} do not match the provided dimensions {h_input}x{w_input}")
    
    # Check if the matrix can be split without cropping
    if h % h_out != 0 or w % w_out != 0:
        # Cropping is needed to ensure the matrix can be split
        matrix = crop_matrix_to_fit(matrix, h_out, w_out, crop_direction)

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create patches and save them as .mat files
    create_patches(matrix, h_out, w_out, input_file_path, output_folder)

def crop_matrix_to_fit(matrix, h_out, w_out, crop_direction):
    """
    Crop the matrix so that it can be evenly split into patches of size (h_out, w_out).
    Uses the specified crop direction or a fallback strategy if necessary.
    """
    h, w, c = matrix.shape

    # Determine the required crop size
    h_needed = (h // h_out) * h_out
    w_needed = (w // w_out) * w_out

    h_excess = h - h_needed
    w_excess = w - w_needed

    # Apply the cropping based on the crop direction and fallback strategy
    if h_excess > 0 or w_excess > 0:
        matrix = apply_cropping(matrix, h_excess, w_excess, crop_direction)

    return matrix

def apply_cropping(matrix, h_excess, w_excess, crop_direction):
    """
    Crops the matrix by the excess amount from the specified direction, with fallback if necessary.
    """
    h, w, c = matrix.shape

    if crop_direction == 'left':
        matrix = matrix[:, w_excess:, :]
    elif crop_direction == 'right':
        matrix = matrix[:, :w - w_excess, :]
    elif crop_direction == 'up':
        matrix = matrix[h_excess:, :, :]
    elif crop_direction == 'down':
        matrix = matrix[:h - h_excess, :, :]
    
    # Fallback strategy: If user direction fails, crop from a different axis
    if matrix.shape[0] > h - h_excess:
        matrix = matrix[:h - h_excess, :, :]  # Crop from the bottom if vertical excess remains
    if matrix.shape[1] > w - w_excess:
        matrix = matrix[:, :w - w_excess, :]  # Crop from the right if horizontal excess remains

    return matrix

def create_patches(matrix, h_out, w_out, input_file_path, output_folder):
    """
    Split the matrix into patches and save each patch as a .mat file.
    """
    h, w, c = matrix.shape
    file_basename = os.path.splitext(os.path.basename(input_file_path))[0]
    
    # Generate patches
    for i in range(0, h, h_out):
        for j in range(0, w, w_out):
            patch = matrix[i:i + h_out, j:j + w_out, :]
            
            # Ensure the patch is of the correct size (may need adjustment at the edges)
            patch = adjust_patch_size(patch, h_out, w_out)

            # Create a filename for the patch and save
            patch_filename = f"{file_basename}_{i}_{j}.mat"
            patch_filepath = os.path.join(output_folder, patch_filename)

            # Save the patch as a .mat file with variable name 'data'
            sio.savemat(patch_filepath, {'data': patch})

def adjust_patch_size(patch, h_out, w_out):
    """
    Adjusts the patch size to h_out x w_out by padding if necessary.
    """
    h_patch, w_patch, c_patch = patch.shape

    if h_patch < h_out or w_patch < w_out:
        new_patch = np.zeros((h_out, w_out, c_patch))
        new_patch[:h_patch, :w_patch, :] = patch
        patch = new_patch

    return patch

if __name__ == '__main__':
    # Example Usage
    input_file_path = 'train_timelapse_256_patches/gualtar_1944/gualtar_1944.mat'
    h_input = 1024
    w_input = 1344
    h_out = 256
    w_out = 256
    crop_direction = 'left'  # or 'right', 'up', 'down'
    output_folder = 'train_timelapse_256_patches/gualtar_1944'

    split_matrix_to_patches(input_file_path, h_input, w_input, h_out, w_out, crop_direction, output_folder)
