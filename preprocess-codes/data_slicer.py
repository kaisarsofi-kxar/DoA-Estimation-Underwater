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
    Input_matrix = mat_data['Input_matrix']

    for i in range(0,101):
        Input_data = np.zeros((12, 12, 2))
        magnitude = np.max(np.abs(Input_matrix[i]))
        real_part = np.real(Input_matrix[i]) / magnitude
        imag_part = np.imag(Input_matrix[i]) / magnitude
        Input_data[:, :, 0] = real_part.astype(np.float64)
        Input_data[:, :, 1] = imag_part.astype(np.float64)
        file_name = f'slice_{i + start_index}.npy'
        file_path = os.path.join(folder_path, file_name)
        np.save(file_path, Input_data)
        print(f"saved slice{i + start_index}")
       
    del Input_matrix
    del mat_data

if __name__ == "__main__":
    sour = [0,5,10];
    
    for range_val in sour:
   
        folder_path = f'/Users/kaisarsofi/Documents/MATLAB/data_gen_code/Murtiza40_60/test_data_samples/increase_angle_both_sources/snr{range_val}/input_samples_source_var_angle'
        
        if not os.path.exists(folder_path):
            # Create the folder
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        mat_files_folder = f'/Users/kaisarsofi/Documents/MATLAB/data_gen_code/Murtiza40_60/test_data/increase_angle_both_sources/snr{range_val}/source_var_angle.mat'
       
       
        # for file in os.listdir(mat_files_folder):
        #     if file.endswith('.mat'):
        #         mat_files.append(os.path.join(mat_files_folder, file))
                
        # mat_files.sort()
        
        # folder_path1 = '/Users/kaisarsofi/Documents/MATLAB/input_input'
               
        start_index = 0
        # for mat_file in mat_files:
        print(mat_files_folder)
        process_mat_file(mat_files_folder, start_index, folder_path)
        # print("range: -",range_val)
        
        # data= np.load(f"{folder_path}/slice_1.npy")
        
        
        # data1=np.load(f"{folder_path1}/slice_1.npy")
    

    
    
    
