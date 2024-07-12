

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:26:13 2024
@author: kaisarsofi
"""

import numpy as np
import os
import mat73

def process_mat_file(mat_files_folder, start_index, folder_path):
    mat_data = mat73.loadmat(mat_files_folder)
    Input_matrix = mat_data['DOA_Labels']

    for i in range(0,200):
        Input_doa = np.zeros((180,))
        Input_doa = Input_matrix[i]
        file_name = f'slice_{i + start_index}.npy'
        file_path = os.path.join(folder_path, file_name)
        np.save(file_path, Input_doa)
        print(f"saved slice{i + start_index}")
       
    del Input_matrix
    del mat_data

if __name__ == "__main__":
    sour = [42,44,46,48,50,52,54,56,58,60];
    
    for range_val in sour:
   
        folder_path = f'/Users/kaisarsofi/Documents/MATLAB/data_gen_code/Murtiza40_60/test_data_samples/dec_angle_source2_60_42/snr5/doa_samples_source_2_at_{range_val}'
        
        if not os.path.exists(folder_path):
            # Create the folder
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        mat_files_folder = f'/Users/kaisarsofi/Documents/MATLAB/data_gen_code/Murtiza40_60/test_data/dec_angle_source2_60_42/snr5/source_2_at_{range_val}.mat'
       
       
        # for file in os.listdir(mat_files_folder):
        #     if file.endswith('.mat'):
        #         mat_files.append(os.path.join(mat_files_folder, file))
                
        # mat_files.sort()
        
        # folder_path1 = '/Users/kaisarsofi/Documents/MATLAB/input_input'
               
        start_index = 0
        # for mat_file in mat_files:
        print(mat_files_folder)
        process_mat_file(mat_files_folder, start_index, folder_path)
    
