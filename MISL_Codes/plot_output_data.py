#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:47:13 2024

@author: kaisarsofi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:58:32 2024

@author: kaisarsofi
"""

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import SubsetRandomSampler, Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import spectrogram





device = "cuda:1" if torch.cuda.is_available() else "cpu"
print("Using device {}".format(device))


def plot_spectrogram(data, title, fs=4000, channel=0):
    plt.figure(figsize=(10, 4))
    sensor_data = data[channel, :, 0]  # Select the specified channel and the first sensor
    nperseg = min(256, sensor_data.shape[0])  # Set nperseg to the minimum of 256 or the input length
    noverlap = max(0, nperseg - 1)   # Ensure noverlap is less than nperseg
    f, t, Sxx = spectrogram(sensor_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'{title} - Sensor 1, Channel {channel + 1}')
    plt.tight_layout()
    plt.show()
class CovarianceLayer(nn.Module):
    
    def __init__(self):
        super(CovarianceLayer, self).__init__()
        
    def forward(self, x):
        cov_mat = []
        
        for row in x:
            res = []
            for channels in range(x.shape[1]):
                res.append(torch.cov(row[channels, :, :].T).unsqueeze(0))
        
            cov_mat.append(torch.cat(res, dim = 0).unsqueeze(0))
        
        return torch.cat(cov_mat)
    
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
    
    
data_folder = "/Users/kaisarsofi/Documents/MATLAB"
input_folder=os.path.join(data_folder,"input_input")
label_folder1=os.path.join(data_folder,"clean_clean")

input_files = [os.path.join(input_folder,file) for file in os.listdir(input_folder)]
label_files1 =[os.path.join(label_folder1,file) for file in os.listdir(label_folder1)]

specific_index = 8000  # Change this to select a different slice

# Load the test dataset without DataLoader for a single slice
input_data = np.load(input_files[specific_index])
label_data1 = np.load(label_files1[specific_index])

# Convert to PyTorch tensors and move to device
input_tensor = torch.tensor(input_data).transpose(0, 2).transpose(1, 2).to(device).unsqueeze(0)  # Add batch dimension
label_tensor = torch.tensor(label_data1).transpose(0, 2).transpose(1, 2).to(device).unsqueeze(0)  # Add batch dimension

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_files, label_files1):
        self.input_files = input_files
        self.label_files1 = label_files1

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_data = np.load(self.input_files[idx])
        label_data1 = np.load(self.label_files1[idx])
        
        input_data = torch.tensor(input_data).transpose(0, 2).transpose(1, 2).to(device)
        label_data1 = torch.tensor(label_data1).transpose(0, 2).transpose(1, 2).to(device)

        # input_data = torch.tensor(input_data).permute(2, 0, 1).to(device)  
        # label_data1 = torch.tensor(label_data1).permute(2, 0, 1).to(device)

        return input_data, label_data1


model = Autoencoder()


PATH="/Users/kaisarsofi/Documents/MATLAB"

state_dict = torch.load(os.path.join(PATH, 'resnet34_autoencoder_best_weights.pth'))

model.load_state_dict(state_dict['state_dict'])
model.to(device)  # Ensure model is on the correct device (GPU or CPU)



batch_size =20
# Define your test dataset and dataloader
test_dataset = CustomDataset(input_files[1950:2000], label_files1[1950:2000])  # Assuming remaining files are for testing
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Set model to evaluation mode
model.eval()



# Forward pass for the specific slice
with torch.no_grad():
    output_tensor = model(input_tensor.float())

# Move data to CPU and convert to numpy for visualization
input_np = input_tensor.cpu().numpy()[0]  # Remove batch dimension
output_np = output_tensor.cpu().numpy()[0]  # Remove batch dimension
label_np = label_tensor.cpu().numpy()[0]  # Remove batch dimension

# Plot spectrograms for the selected slice
plot_spectrogram(input_np, title='Input')
plot_spectrogram(output_np, title='Output')
plot_spectrogram(label_np, title='Label')






plt.figure(figsize=(10, 4))
sensor_data = input_np[1, :, 0]  # Select the specified channel and the first sensor
nperseg = min(256, sensor_data.shape[0])  # Set nperseg to the minimum of 256 or the input length
noverlap = max(0, nperseg - 1)   # Ensure noverlap is less than nperseg
f, t, Sxx = spectrogram(sensor_data, fs=4000, nperseg=nperseg, noverlap=noverlap)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('INput - Sensor 1, Channel 0')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 4))
sensor_data = output_np[1, :, 0]  # Select the specified channel and the first sensor
nperseg = min(256, sensor_data.shape[0])  # Set nperseg to the minimum of 256 or the input length
noverlap = max(0, nperseg - 1)   # Ensure noverlap is less than nperseg
f, t, Sxx = spectrogram(sensor_data, fs=4000, nperseg=nperseg, noverlap=noverlap)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('OUtput - Sensor 1, Channel 0')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 4))
sensor_data = label_np[1, :, 0]  # Select the specified channel and the first sensor
nperseg = min(256, sensor_data.shape[0])  # Set nperseg to the minimum of 256 or the input length
noverlap = max(0, nperseg - 1)   # Ensure noverlap is less than nperseg
f, t, Sxx = spectrogram(sensor_data, fs=4000, nperseg=nperseg, noverlap=noverlap)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('LAbel - Sensor 1, Channel 0')
plt.tight_layout()
plt.show()


# # Define criterion for evaluation (MSE Loss)
# criterion = nn.MSELoss()

# # Initialize variables for tracking test loss
# test_loss = 0.0
# num_batches = 0

# # Iterate over test dataset with tqdm
# with tqdm(total=len(test_loader), desc='Testing') as pbar:
#     for inputs, labels1 in test_loader:
#         inputs, labels1 = inputs.to(device), labels1.to(device)
        
#         # Forward pass
#         outputs1 = model(inputs.float())
        
#         # Calculate loss
#         batch_loss = criterion(outputs1, labels1.float())
#         test_loss += batch_loss.item()
#         num_batches += 1
        
#         # Update progress bar
#         pbar.update(1)

# # Calculate average test loss
# avg_test_loss = test_loss / num_batches
# print(f'\nAverage Test MSE: {avg_test_loss:.4f}')

# results_path = os.path.join(PATH, 'test_results.txt')
# with open(results_path, 'w') as f:
#     f.write(f'Average Test Loss: {avg_test_loss:.4f}\n')
