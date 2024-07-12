#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:26:51 2024

@author: kaisarsofi
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import SubsetRandomSampler, Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F



device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device {}".format(device))

      
class DirectDoaModel(nn.Module):
    def __init__(self):
        super(DirectDoaModel, self).__init__()
        # Define layers
        self.dropout= nn.Dropout(0.3)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(256 * 4 * 4, 2048)
        self.fc2 = nn.Linear(2048, 1024) 
        self.fc3 = nn.Linear(1024, 180)


    def forward(self, x):
        # Apply convolutions and batch normalization without pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))  # Output layer with sigmoid for binary classification
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.dropout= nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.initial = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=3,stride= 1,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()                
                )
        
        self.max_pool_1 = nn.Sequential( 
              nn.MaxPool2d(kernel_size=3,stride=(1,1),padding=0), #,stride=(2,1)
              )
        
        self.identity_1 = nn.Sequential( nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64)
            )
        
        self.projection_plain_1= nn.Sequential(
            # nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, kernel_size=3,stride=(1,1),padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128)
            )
        
        self.projection_shortcut_1= nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3,stride=1,padding=0)
            )
        
        self.identity_2 = nn.Sequential( nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128)
            )
        
        self.projection_shortcut_2= nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=0)
            )
        
        self.projection_plain_2= nn.Sequential(
            # nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=0),
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
            nn.Conv2d(256, 512, kernel_size=3,stride=2,padding=1)
            )
        
        self.projection_plain_3= nn.Sequential(
            # nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, kernel_size=3,stride=2,padding=1),
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
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=(2, 2), padding=1, output_padding=(0,0 )),
            nn.ReLU(),
            )
        # #256
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=(2, 2), padding=3, output_padding=(0,0 )),  
            nn.ReLU(),
            )
        # #128
        self.d3 = nn.Sequential(  
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=(2, 2), padding=4, output_padding=(0, 0)), 
            nn.ReLU(),
            )
        #64
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=5, output_padding=(0, 0)), 
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
        x1=self.dropout(x1)
        
        # print("x1: ",x1.shape)
        x1= self.max_pool_1(x1)
        # print("x1: ",x1.shape)
        
        #BL-1
        f_x1= self.identity_1(x1)
        f_x1=self.dropout(f_x1)
        # print("f_x1: ",f_x1.shape)
        h_x1= torch.add(x1,f_x1)
        # print("h_x1: ",h_x1.shape)
        h_x1= self.relu(h_x1)
        x2= h_x1
        
        #BL-2
        f_x2= self.identity_1(x2)
        f_x2=self.dropout(f_x2)
        # print("f_x2: ",f_x2.shape)
        h_x2= torch.add(x2,f_x2)
        h_x2= self.relu(h_x2)
        x3=h_x2
        
        #BL-3
        f_x3= self.identity_1(x3)
        f_x3=self.dropout(f_x3)
        h_x3= torch.add(x3,f_x3)
        h_x3= self.relu(h_x3)
        x4=h_x3
        
        #BL-4
        f_x4= self.projection_plain_1(x4)
        f_x4=self.dropout(f_x4)
        # print("f_x4: ",f_x4.shape)
        # print("shourt: ",self.projection_shortcut_1(x4).shape)
        h_x4= torch.add(f_x4,self.projection_shortcut_1(x4))
        h_x4= self.relu(h_x4)
        x5= h_x4
        
        #BL-5
        f_x5= self.identity_2(x5)
        f_x5=self.dropout(f_x5)
        h_x5= torch.add(f_x5,x5)
        h_x5= self.relu(h_x5)
        x6= h_x5
        
        #BL-6
        f_x6= self.identity_2(x6)
        f_x6=self.dropout(f_x6)
        h_x6= torch.add(f_x6,x6)
        h_x5= self.relu(h_x6)
        x7= h_x6
        
        #BL-7
        f_x7= self.identity_2(x7)
        f_x7=self.dropout(f_x7)
        h_x7= torch.add(f_x7,x7)
        h_x7= self.relu(h_x7)
        x8= h_x7
        
        # print("x8: ",x8.shape)
        
        #BL-8
        f_x8= self.projection_plain_2(x8)
        f_x8=self.dropout(f_x8)
        # print("f_x8: ",f_x8.shape)
        # print("shourt1: ",self.projection_shortcut_2(x8).shape)
        h_x8= torch.add(f_x8,self.projection_shortcut_2(x8))
        h_x8= self.relu(h_x8)
        x9= h_x8
        
        #BL- (9-13)
        for i in range(0,5):
            f_x9= self.identity_3(x9)
            f_x9=self.dropout(f_x9)
            h_x9= torch.add(f_x9,x9)
            h_x9= self.relu(h_x9)
            x9= h_x9
        
        x14=x9
        
        # print("x14: ",x14.shape)
        
        #BL-14
        f_x14= self.projection_plain_3(x14)
        f_x14=self.dropout(f_x14)
        # print("f_x14: ",f_x14.shape)
        # print("short3: ",self.projection_shortcut_3(x14).shape)
        h_x14= torch.add(f_x14,self.projection_shortcut_3(x14))
        h_x14= self.relu(h_x14)
        x15= h_x14
        
        #BL- 15
        f_x15= self.identity_4(x15)
        f_x15=self.dropout(f_x15)
        h_x15= torch.add(f_x15,x15)
        h_x15= self.relu(h_x15)
        x16= h_x15
       
        #BL- 16
        f_x16= self.identity_4(x16)
        f_x16=self.dropout(f_x16)
        h_x16= torch.add(f_x16,x16)
        h_x16= self.relu(h_x16)
        x17= h_x16
        
        
        
        #decoder
        d_5= self.d5(x17)
        d_5 = self.dropout(d_5)
        d_4= self.d4(d_5)
        d_4 = self.dropout(d_4)
        d_3= self.d3(d_4)
        d_3 = self.dropout(d_3)
        d_2= self.d2(d_3)
        d_2 = self.dropout(d_2)
        d_1= self.d1(d_2)
        return d_1
    

class CombinedModel(nn.Module):
    def __init__(self, autoencoder, cnn):
        super(CombinedModel, self).__init__()
        self.autoencoder = autoencoder
        self.direct_doa_model = direct_doa_model

    def forward(self, x):
       
        x = self.autoencoder(x)
       
        x = self.direct_doa_model(x)  # Estimate DOA
       
        return x


direct_doa_model = DirectDoaModel()
autoencoder=Autoencoder()

combined_model = CombinedModel(autoencoder, direct_doa_model)


data_folder = "/Users/kaisarsofi/Documents/MATLAB/data_gen_code/Data_gen_covariance/data_gen_variable_sensor_depth"
input_folder=os.path.join(data_folder,"input_samples")
label_folder1=os.path.join(data_folder,"label_samples")
label_folder2=os.path.join(data_folder,"doa_samples")    #doa

input_files = [os.path.join(input_folder,file) for file in os.listdir(input_folder)]
label_files1 =[os.path.join(label_folder1,file) for file in os.listdir(label_folder1)]
label_files2 =[os.path.join(label_folder2,file) for file in os.listdir(label_folder2)]
# input_files.remove('/Users/kaisarsofi/Documents/MATLAB/input_covariance/.DS_Store')

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_files, label_files1, label_files2):
        self.input_files = input_files
        self.label_files1 = label_files1
        self.label_files2 = label_files2

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_data = np.load(self.input_files[idx])
        label_data1 = np.load(self.label_files1[idx])
        label_data2 = np.load(self.label_files2[idx])
        
        input_data = torch.tensor(input_data).transpose(0, 2).transpose(1, 2).to(device)
        label_data1 = torch.tensor(label_data1).transpose(0, 2).transpose(1, 2).to(device)
        label_data2 = torch.tensor(label_data2).to(device)

        # input_data = torch.tensor(input_data).permute(2, 0, 1).to(device)  
        # label_data1 = torch.tensor(label_data1).permute(2, 0, 1).to(device)

        return input_data, label_data1, label_data2

custom_dataset = CustomDataset(input_files, label_files1, label_files2  )

# Define batch size
batch_size = 32
print("batch_size: ",batch_size)

# Split data into training and validation sets
dataset_size = len(custom_dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.8 * dataset_size))  # 80-20 split
np.random.shuffle(indices)
train_indices, val_indices = indices[:split], indices[split:]

# Define samplers for obtaining batches
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Create data loaders for training and validation sets
train_loader = torch.utils.data.DataLoader(custom_dataset, batch_size = batch_size, sampler = train_sampler)
val_loader = torch.utils.data.DataLoader(custom_dataset, batch_size = batch_size, sampler = val_sampler)


# Initialize your model, criterion, optimizer, and scheduler
autoencoder.to(device)
direct_doa_model.to(device)
combined_model.to(device)


criterion_autoencoder = nn.MSELoss()
criterion_direct_doa_model = nn.BCELoss() # Adjust if your task is classification

optimizer = optim.Adam([
    {'params': autoencoder.parameters(), 'lr': 0.0001},
    {'params': direct_doa_model.parameters(), 'lr': 0.00001}
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Set up lists to track training and validation losses
train_loss_autoencoder = []
val_loss_autoencoder = []
train_loss_direct_doa_model = []
val_loss_direct_doa_model = []

# Set up early stopping parameters
best_val_loss = float('inf')
patience = 5
epochs_no_improve = 0
early_stop = False

num_epochs = 50

for epoch in range(num_epochs):
    if early_stop:
        print("Early stopping")
        break

    combined_model.train()
    running_loss_autoencoder = 0.0
    running_loss_direct_doa_model = 0.0
    
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for i, data in enumerate(train_loader, 0):
            inputs, labels1, labels2 = data
            inputs, labels1, labels2 = inputs.to(device), labels1.to(device), labels2.to(device)
    
            optimizer.zero_grad()
            
            outputs1 = autoencoder(inputs.float())
            loss_autoencoder = criterion_autoencoder(outputs1, labels1.float())
            
            outputs2 = direct_doa_model(outputs1)
            loss_direct_doa_model = criterion_direct_doa_model(outputs2, labels2.float())
            
            total_loss = loss_autoencoder + loss_direct_doa_model
    
            total_loss.backward()
            optimizer.step()
    
            running_loss_autoencoder += loss_autoencoder.item()
            running_loss_direct_doa_model += loss_direct_doa_model.item()
            avg_loss_autoencoder = running_loss_autoencoder / (i + 1)
            avg_loss_cnn = running_loss_direct_doa_model / (i + 1)
            pbar.set_postfix({'res34 Loss': avg_loss_autoencoder, 'direct_doa_model Loss': avg_loss_cnn})
            pbar.update()
    
    avg_train_loss_autoencoder = running_loss_autoencoder / len(train_loader)
    avg_train_loss_direct_doa_model = running_loss_direct_doa_model / len(train_loader)
    train_loss_autoencoder.append(avg_train_loss_autoencoder)
    train_loss_direct_doa_model.append(avg_train_loss_direct_doa_model)
    
    # Validation phase
    combined_model.eval()
    val_loss_autoencoder_epoch = 0.0
    val_loss_direct_doa_model_epoch = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels1, labels2 = data
            inputs, labels1, labels2 = inputs.to(device), labels1.to(device), labels2.to(device)
            
            outputs1 = autoencoder(inputs.float())
            val_loss_autoencoder_epoch += criterion_autoencoder(outputs1, labels1.float()).item()
            
            
            outputs2 = direct_doa_model(outputs1)
            val_loss_direct_doa_model_epoch += criterion_direct_doa_model(outputs2, labels2.float()).item()
    
    avg_val_loss_autoencoder = val_loss_autoencoder_epoch / len(val_loader)
    avg_val_loss_direct_doa_model = val_loss_direct_doa_model_epoch / len(val_loader)
    val_loss_autoencoder.append(avg_val_loss_autoencoder)
    val_loss_direct_doa_model.append(avg_val_loss_direct_doa_model)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] - Training Res34 Loss: {avg_train_loss_autoencoder:.4f}, Validation Res34 Loss: {avg_val_loss_autoencoder:.4f}, Training direct_doa_model Loss: {avg_train_loss_direct_doa_model:.4f}, Validation direct_doa_model Loss: {avg_val_loss_direct_doa_model:.4f}')
    
    # Update scheduler
    scheduler.step()
    
    # Check for early stopping
    if avg_val_loss_autoencoder + avg_val_loss_direct_doa_model < best_val_loss:
        best_val_loss = avg_val_loss_autoencoder + avg_val_loss_direct_doa_model
        torch.save(combined_model.state_dict(), 'combined_res34_doa_model_diff_LR.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            early_stop = True

print('Training complete.')

# Plot the losses
plt.figure(figsize=(10, 8))
plt.plot(train_loss_autoencoder, label='Training Autoencoder Loss')
plt.plot(val_loss_autoencoder, label='Validation Autoencoder Loss')
plt.plot(train_loss_direct_doa_model, label='Training CNN Loss')
plt.plot(val_loss_direct_doa_model, label='Validation CNN Loss')
plt.legend()
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()