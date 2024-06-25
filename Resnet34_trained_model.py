import numpy as np
from scipy.io import loadmat, savemat
import os
import mat73
import scipy.io
import h5py
import torch
import torch.nn as nn
import sys
from scipy.io import loadmat, savemat
import ast

x=[]
mat_data = ast.literal_eval(sys.argv[1])
# mat_file_path = '/DATA1/MURTIZA/Gautam_and_Randhir_codes/data11_new.mat'
# mat_data = mat73.loadmat(mat_file_path)

folder_path = '/DATA1/MURTIZA/Gautam_and_Randhir_codes/noisy_test_input_matrix'

Input_data = np.zeros((4000, 8, 2))
#Input_matrix = mat_data['Input_matrix']
Input_matrix = mat_data[0,:,:]
for i in range(0,1):
    #Input_data = np.zeros((4000, 8, 2))
    magnitude = np.max(np.abs(Input_matrix))
    x.append(magnitude)
    magnitude = np.max(np.abs(Input_matrix))
    real_part = np.real(Input_matrix)/magnitude
    imag_part = np.imag(Input_matrix)/magnitude
    #phase_part = np.angle(Input_matrix[i])
    Input_data[:, :, 0] = real_part.astype(np.float64)
    Input_data[ :, :, 1] = imag_part.astype(np.float64)
    del real_part,imag_part


device = "cuda:0" if torch.cuda.is_available() else "cpu"
PATH="/DATA1/MURTIZA/Gautam_and_Randhir_codes"
#creating trace of my model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.initial = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=7,stride= (1,1),padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU()                
                )
        
        self.max_pool_1 = nn.Sequential( 
             nn.MaxPool2d(kernel_size=(2,1),stride=(2,1)), #,stride=(2,1)
             )
        
        self.identity_1 = nn.Sequential( nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64)
            )
        
        self.projection_plain_1= nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, kernel_size=3,stride=(2,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128)
            )
        
        self.projection_shortcut_1= nn.Sequential(
            nn.Conv2d(64,128, kernel_size= 1,stride=(2,1))
            )
        
        self.identity_2 = nn.Sequential( nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128)
            )
        
        self.projection_shortcut_2= nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1,stride=(2,1))
            )
        
        self.projection_plain_2= nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=3,stride=(2,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256)
            )
        
        self.identity_3=nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(256)
            )
        
        self.projection_shortcut_3= nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1,stride=(2,1))
            )
        
        self.projection_plain_3= nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, kernel_size=3,stride=(2,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512)
            )
        
        self.identity_4= nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(512)
            )
        
        #decoder
        self.d5 = nn.Sequential( 
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            )
        # #256
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)),  
            nn.ReLU(),
            )
        # #128
        self.d3 = nn.Sequential(  
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)), 
            nn.ReLU(),
            )
        #64
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)), 
            nn.ReLU(),
            )
        self.d1 = nn.Sequential(
              nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )


      
    
            
    def forward(self, x):
        
        #encoder
        #x = x.unsqueeze(0)
        x1= self.initial(x)
        x1= self.max_pool_1(x1)
        
        #BL-1
        f_x1= self.identity_1(x1)
        h_x1= torch.add(x1,f_x1)
        h_x1= nn.ReLU()(h_x1)
        x2= h_x1
        
        #BL-2
        f_x2= self.identity_1(x2)
        h_x2= torch.add(x2,f_x2)
        h_x2= nn.ReLU()(h_x2)
        x3=h_x2
        
        #BL-3
        f_x3= self.identity_1(x3)
        h_x3= torch.add(x3,f_x3)
        h_x3= nn.ReLU()(h_x3)
        x4=h_x3
        
        #BL-4
        f_x4= self.projection_plain_1(x4)
        # c=self.projection_shortcut_1(x4)
        h_x4= torch.add(f_x4,self.projection_shortcut_1(x4))
        h_x4= nn.ReLU()(h_x4)
        x5= h_x4
        
        #BL-5
        f_x5= self.identity_2(x5)
        h_x5= torch.add(f_x5,x5)
        h_x5= nn.ReLU()(h_x5)
        x6= h_x5
        
        #BL-6
        f_x6= self.identity_2(x6)
        h_x6= torch.add(f_x6,x6)
        h_x5= nn.ReLU()(h_x6)
        x7= h_x6
        
        #BL-7
        f_x7= self.identity_2(x7)
        h_x7= torch.add(f_x7,x7)
        h_x7= nn.ReLU()(h_x7)
        x8= h_x7
        
        #BL-8
        f_x8= self.projection_plain_2(x8)
        h_x8= torch.add(f_x8,self.projection_shortcut_2(x8))
        h_x8= nn.ReLU()(h_x8)
        x9= h_x8
        
        #BL- (9-13)
        for i in range(0,5):
            f_x9= self.identity_3(x9)
            h_x9= torch.add(f_x9,x9)
            h_x9= nn.ReLU()(h_x9)
            x9= h_x9
        
        x14=x9
        
        #BL-14
        f_x14= self.projection_plain_3(x14)
        h_x14= torch.add(f_x14,self.projection_shortcut_3(x14))
        h_x14= nn.ReLU()(h_x14)
        x15= h_x14
        
        #BL- 15
        f_x15= self.identity_4(x15)
        h_x15= torch.add(f_x15,x15)
        h_x15= nn.ReLU()(h_x15)
        x16= h_x15
       
        #BL- 16
        f_x16= self.identity_4(x16)
        h_x16= torch.add(f_x16,x16)
        h_x16= nn.ReLU()(h_x16)
        x17= h_x16
        
        
        
        #decoder
        d_5= self.d5(x17)
        d_4= self.d4(d_5)
        d_3= self.d3(d_4)
        d_2= self.d2(d_3)
        d_1= self.d1(d_2)
        return d_1
     
        
autoencoder = Autoencoder()

# state_dict = torch.load(os.path.join(PATH,'ResNet34_model_weights.pth'))

# # Handle 'module' prefix if present in the keys
# if list(state_dict.keys())[0].startswith('module'):
#     state_dict = {k[7:]: v for k, v in state_dict.items()}




# Load the PyTorch model weights
state_dict = torch.load(os.path.join(PATH, 'ResNet34_model_weights.pth'))

# Convert weights and bias parameters to np.float64
for k, v in state_dict.items():
    # Check if the parameter is bias or weights
    state_dict[k] = v.to(torch.float64)

# Handle 'module' prefix if present in the keys
if list(state_dict.keys())[0].startswith('module'):
    state_dict = {k[7:]: v for k, v in state_dict.items()}

# i=0
# for k, v in state_dict.items():
#     # Check if the parameter is bias or weights
#     if v.is_cuda:
#         v = v.cpu()
#     print(k," ",v.dtype)
#     if i==1:
#         break
#     i=i+1
    

# i=0 
# file_name = f'slice_{i}.npy'
# file_path = os.path.join(folder_path, file_name)
# Input_data = np.load(file_path)
# input_data = torch.tensor(Input_data).permute(2, 0, 1).to(device) 
# print("datatype_of_input is ",input_data.dtype)

# #%%
# import os
# import torch
# data_folder = "/DATA1/MURTIZA"
# input_folder=os.path.join(data_folder,"input_input_input1")
# # label_folder1=os.path.join(data_folder,"clean_clean_clean1")
# # label_folder2=os.path.join(data_folder,"doa_doa_doa1")

# input_files=os.path.join(input_folder,'slice_0.npy')
# input_data = np.load(input_files)
# # label_data1 = np.load(self.label_files1[idx])
# # label_data2 = np.load(self.label_files2[idx])
    
#    #index_to_remove = 2  # Index of the element to remove
   
# input_data = np.delete(input_data, 2, axis=-1)
# #label_data1 = np.delete(label_data1, 2, axis=-1)
#    # input_data = torch.tensor(input_data).permute(2, 0, 1)  
#     # label_data1 = torch.tensor(label_data1).permute(2, 0, 1)  
# input_data = torch.tensor(input_data).permute(2, 0, 1).to(device)  
# print(input_data.dtype)

# i=0
# folder_path = '/DATA1/MURTIZA/Gautam_and_Randhir_codes/noisy_test_input_matrix'
# file_name = f'slice_{i}.npy'
# file_path = os.path.join(folder_path, file_name)
# # Input_data = np.zeros((4000, 8, 3))
# # magnitude = np.max(np.abs(Input))
# # real_part = np.real(Input)/magnitude
# # imag_part = np.imag(Input)/magnitude
# # Input_data[:, :, 0] = real_part.astype(np.float64)
# # Input_data[ :, :, 1] = imag_part.astype(np.float64)
# # del real_part,imag_part
# Input_data = np.load(file_path)
# input_data = torch.tensor(Input_data).permute(2, 0, 1).to(device).float() 
# print(Input_data.shape)



from scipy import io
import h5py
#import matlab.engine
# eng = matlab.engine.start_matlab()
#shape = (1000,4000,8)
shape = (8,4000)
arr=np.empty(shape=shape, dtype=complex)

autoencoder.to(device)

for i in range(0,1):
    
    # Load weights
    autoencoder.load_state_dict(state_dict)
    
    # Set to evaluation model
    autoencoder.eval()
    # file_name = f'slice_{i}.npy'
    # file_path = os.path.join(folder_path, file_name)
    # # Input_data = np.zeros((4000, 8, 3))
    # # magnitude = np.max(np.abs(Input))
    # real_part = np.real(Input)/magnitude
    # imag_part = np.imag(Input)/magnitude
    # Input_data[:, :, 0] = real_part.astype(np.float64)
    # Input_data[ :, :, 1] = imag_part.astype(np.float64)
    # del real_part,imag_part
    #Input_data = np.load(file_path)
    input_data = torch.tensor(Input_data).permute(2, 0, 1).to(device).float() 
    input_data=input_data.unsqueeze(0)
    output = autoencoder(input_data)
    del input_data
    #print(output.shape)
    output=output.squeeze(0)
    #print(output.shape)

    magnitude=x[i]
    output=(output*magnitude)
    output=output.permute(1,2,0)
    # Print or use the output as needed
    output_cpu = output.cpu()
    output_cpu=output_cpu.detach().numpy()
    
    
    # Load the NumPy array from .npy file
    numpy_data = output_cpu
    #print(numpy_data.shape)
    del output_cpu
    real_part = numpy_data[:,:,0]  # (4000, 8) array
    imaginary_part = numpy_data[:,:,1]  # (4000, 8) array

    # Create complex numbers
    numpy_data = real_part + 1j * imaginary_part
    del real_part, imaginary_part
    #print(numpy_data.shape)
    numpy_data=numpy_data.reshape(8,4000)
    arr[:,:]=numpy_data
    del numpy_data


   
    PAT="/DATA1/MURTIZA/Gautam_and_Randhir_codes/cleaned_matrix1.mat"
    
    # Load the original .mat file
    data_dict = loadmat('/DATA1/MURTIZA/Gautam_and_Randhir_codes/cleaned_matrix1.mat')

    # # Load the new .mat file
    # new_data_dict = loadmat('new_data.mat')

    # Replace the data field
    data_dict['data'] = arr

    # Save the modified .mat file
    savemat('/DATA1/MURTIZA/Gautam_and_Randhir_codes/cleaned_matrix1.mat', data_dict)


    #save .mat file using engine
    #io.savemat(PAT, {'data': arr})
    
    del arr
    # Create the HDF5 file and write the data
    # with h5py.File(f_path, 'w') as f:
    #     f.create_dataset('data', data=numpy_data)

#%%
