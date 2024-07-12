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


Angles = -90:1:90-1; SNR = 0;
range = 7000;
sensor_depth = -50;
output_folder= '/Users/kaisarsofi/Documents/MATLAB/data_gen_code/Murtiza40_60/test_data/range_snr0';

if ~exist(output_folder, 'dir')
    % Create the folder if it doesn't exist
    mkdir(output_folder);
    disp(['Created folder: ', output_folder]);
end
val=40;
mat_filename = sprintf('new_data40_60_range%d.mat',abs(range));

output_filename = fullfile(output_folder, mat_filename);

%% Source sound characteristics

% musicspatialspect = phased.MUSICEstimator('SensorArray',array,'PropagationSpeed',propSpeed,'OperatingFrequency',...

%    OperatingFrequency,'ForwardBackwardAveraging',false,'ScanAngles',-90:1:90-1,'DOAOutputPort',true,'NumSignalsSource','Property','NumSignals',sources);%'SpatialSmoothing',5

%%  DOA estimation using Inbuilt MUSIC Alogrithm

PP = length(Angles);fs = 4000;

duration = 1;

t = linspace(0,duration,fs*duration);

sound1 = cos(2*pi*1000*t)';

sound2 = cos(2*pi*1500*t)';

% DOA estimation using Inbuilt MUSIC Algorithm

Samples = 1000;

DOA_Labels =[];

Input_matrix = [];

Label_matrix = [];

sam_num=1;
for pp = 1:length(sensor_depth)

    for jj =1:length(range)

        for k=1:length(SNR)

            Y_training = zeros (Samples,numberofSensors,numberofSensors);

            Y_label = zeros (Samples,numberofSensors,numberofSensors);

            Binary_DOAs1= zeros(Samples,length(Angles));

          

            for i = 1:Samples

                disp(sam_num);

          

                % randomIndices = randperm(length(Angles), sources);
                
                % randomThetas = Angles(randomIndices);

                randomThetas = [40,60];
                binary_vector = zeros(size(Angles));

              

                for ii = 1:length(randomThetas)

                    idx = find(Angles == randomThetas(ii));

                    binary_vector(idx) = 1;

                end

                Binary_DOAs1(i,:) = binary_vector;

                x1 = range(jj)*sind(90)*cosd(randomThetas(1));  y1 = range(jj)*sind(90)*sind(randomThetas(1));  z1 = range(jj)*cosd(90);

                x2 = range(jj)*sind(90)*cosd(randomThetas(2));  y2 = range(jj)*sind(90)*sind(randomThetas(2));  z2 = range(jj)*cosd(90);

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
               


                Y_training(i,:,:)  = 1/size(Mixed_signal,1)*(Mixed_signal' * Mixed_signal);




 

                [paths1,dop1,aloss1,rcvang1,srcang1] = isopaths2(pos_tx1,pos_rx,vel_tx1,vel_rx,duration); 

                [paths2,dop2,aloss2,rcvang2,srcang2] = isopaths2(pos_tx2,pos_rx,vel_tx2,vel_rx,duration); 

                tsig1 = projRadiator(sound1,srcang1);

                rsig1 = channel3(tsig1,paths1,dop1,aloss1);

                tsig2 = projRadiator(sound2,srcang2);

                rsig2 = channel4(tsig2,paths2,dop2,aloss2);

                recieved_signal1 = arrayCollector(rsig1,rcvang1);

                recieved_signal2 = arrayCollector(rsig2,rcvang2);



               

                Label_mixed = recieved_signal1 + recieved_signal2;

                Y_label(i,:,:) = 1/size(Label_mixed,1)*(Label_mixed' * Label_mixed);

                sam_num=sam_num+1;

            end

            Input_matrix = [Input_matrix; Y_training];

            Label_matrix = [Label_matrix; Y_label];

            DOA_Labels = [DOA_Labels; Binary_DOAs1];

        end

    end

end

save(output_filename, 'Input_matrix', 'Label_matrix', 'DOA_Labels','-v7.3');
disp('saved: ');
disp(output_filename);