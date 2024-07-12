%% Intilization
clear all; close all; clc; propSpeed = 1520; channelDepth = 120; OperatingFrequency = 4000; numberofSensors = 12; sources = 2;
 new_frequencies = 100:100:10000; num_paths_inputs = 51;
channel1 = phased.MultipathChannel('OperatingFrequency',OperatingFrequency);
channel2 = phased.MultipathChannel('OperatingFrequency',OperatingFrequency);      
channel3 = phased.MultipathChannel('OperatingFrequency',OperatingFrequency);
channel4 = phased.MultipathChannel('OperatingFrequency',OperatingFrequency);
isopaths1 = phased.IsoSpeedUnderwaterPaths('ChannelDepth',channelDepth,'BottomLoss',0.2,...
  'NumPathsSource','Property','NumPaths',num_paths_inputs,'PropagationSpeed',propSpeed,'LossFrequencies', new_frequencies);
isopaths2 = phased.IsoSpeedUnderwaterPaths('ChannelDepth',channelDepth,'BottomLoss',0.2,...
  'NumPathsSource','Property','NumPaths',1,'PropagationSpeed',propSpeed,'LossFrequencies', new_frequencies);
projector = phased.IsotropicProjector('VoltageResponse',120);
projRadiator = phased.Radiator('Sensor',projector,'PropagationSpeed',propSpeed,'OperatingFrequency',OperatingFrequency);
hydrophone = phased.IsotropicHydrophone('VoltageSensitivity',-150);
array = phased.ULA('Element',hydrophone,'NumElements',numberofSensors,'ElementSpacing',propSpeed/OperatingFrequency/2,'ArrayAxis','y');
arrayCollector = phased.Collector('Sensor',array,'PropagationSpeed',propSpeed,'OperatingFrequency',OperatingFrequency);
Angles = -90:1:90-1; SNR = -10:5:10;
range = 1000:1000:5000;
sensor_depth = -20:-20:-100;

%%  DOA estimation using Inbuilt MUSIC Alogrithm
PP = length(Angles);fs = 5000;
duration = 1;
t = linspace(0,duration,fs*duration);
sound1 = cos(2*pi*1000*t)';
sound2 = cos(2*pi*2000*t)';
% DOA estimation using Inbuilt MUSIC Algorithm
DOA_Labels =[];
Input_matrix = [];
Label_matrix = [];
sam_num=1;
Learning_angles=-60:2:60;%-60:2:60;
[angle1, angle2] = ndgrid(Learning_angles, Learning_angles);
combinations = [angle1(:), angle2(:)];
unique_combinations = combinations(combinations(:,1) <= combinations(:,2), :);
unique_combinations([1, end], :) = [];


for i = 1:length(unique_combinations)
                binary_vector = zeros(size(Angles));
                for ii = 1:sources
                    idx = find(Angles == unique_combinations(i,ii));
                    binary_vector(idx) = 1;
                end
      for pp = 1:length(sensor_depth)
        for jj =1:length(range)
          for k=1:length(SNR)
                x1 = range(jj)*sind(90)*cosd(unique_combinations(i,1));  y1 = range(jj)*sind(90)*sind(unique_combinations(i,1));  z1 = range(jj)*cosd(90);
                x2 = range(jj)*sind(90)*cosd(unique_combinations(i,2));  y2 = range(jj)*sind(90)*sind(unique_combinations(i,2));  z2 = range(jj)*cosd(90);
                beaconPlatform1 = phased.Platform('InitialPosition',[x1; y1; sensor_depth(pp) + z1],'Velocity',[0; 0; 0]);
                beaconPlatform2 = phased.Platform('InitialPosition',[x2; y2; sensor_depth(pp) + z2],'Velocity',[0; 0; 0]);
                arrayPlatform1 = phased.Platform('InitialPosition',[0; 0; sensor_depth(pp)],'Velocity',[0; 0; 0]);
                [pos_tx1,vel_tx1] = beaconPlatform1(duration);      % Update acoustic beacon 1 positions
                [pos_rx,vel_rx] = arrayPlatform1(duration);          % Update array positions
                [pos_tx2,vel_tx2] = beaconPlatform2(duration);      % Update acoustic beacon 1 positions
                [paths1,dop1,aloss1,rcvang1,srcang1] = isopaths1(pos_tx1,pos_rx,vel_tx1,vel_rx,duration);  
                [paths2,dop2,aloss2,rcvang2,srcang2] = isopaths1(pos_tx2,pos_rx,vel_tx2,vel_rx,duration);  
                tsig1 = projRadiator(sound1,srcang1);
                rsig1 = channel1(tsig1,paths1,dop1,aloss1);
                tsig2 = projRadiator(sound2,srcang2);
                rsig2 = channel2(tsig2,paths2,dop2,aloss2);
                recieved_signal1 = arrayCollector(rsig1,rcvang1);
                recieved_signal2 = arrayCollector(rsig2,rcvang2);
                combined_signal = recieved_signal1 + recieved_signal2;
                squared_magnitude =  mean(abs(combined_signal).^2,1);
                signal_power = sum(squared_magnitude);
                noise_power = signal_power / (10^(SNR(k)/10));
                noise = zeros(numberofSensors,size(recieved_signal2,1));
                for iii = 1:size(recieved_signal1,1)
                    noise(:,iii) = sqrt(noise_power)/2 .* (randn(numberofSensors,1) + 1i*randn(numberofSensors,1));
                end
                noise = noise';
                Mixed_signal =  combined_signal + noise;
                Y_training  = 1/size(Mixed_signal,1)*(Mixed_signal' * Mixed_signal);
                disp(sam_num)

                [paths1,dop1,aloss1,rcvang1,srcang1] = isopaths2(pos_tx1,pos_rx,vel_tx1,vel_rx,duration);  
                [paths2,dop2,aloss2,rcvang2,srcang2] = isopaths2(pos_tx2,pos_rx,vel_tx2,vel_rx,duration);  
                tsig1 = projRadiator(sound1,srcang1);
                rsig1 = channel3(tsig1,paths1,dop1,aloss1);
                tsig2 = projRadiator(sound2,srcang2);
                rsig2 = channel4(tsig2,paths2,dop2,aloss2);
                recieved_signal1 = arrayCollector(rsig1,rcvang1);
                recieved_signal2 = arrayCollector(rsig2,rcvang2);
                Label_mixed = recieved_signal1 + recieved_signal2;
                Y_label = 1/size(Label_mixed,1)*(Label_mixed' * Label_mixed);

                filename = sprintf('file_i%d_pp%d_jj%d_k%d.mat', i, pp, jj, k);
                save(fullfile('./Data_gen/Input', filename), 'Y_training');
                save(fullfile('./Data_gen/Label', filename), 'Y_label');
                save(fullfile('./Data_gen/DOA', filename), 'binary_vector');
                sam_num = sam_num+1;

          end
        end
      end
end

