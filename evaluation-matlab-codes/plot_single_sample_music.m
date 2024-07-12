clear all; close all; clc;
propSpeed = 1520; % Speed of sound in water (m/s).
channelDepth = 200; % Depth of the water channel (meters).
OperatingFrequency = 4000; % Operating frequency (Hz).
numberofSensors = 8; % Number of sensors in the array.
sources = 1; % Number of sound sources.
new_frequencies = 100:500:10000; % Range of frequencies for bottom loss calculation.
num_paths_inputs = 51; % Number of paths for the underwater channel.

isopaths1 = phased.IsoSpeedUnderwaterPaths('ChannelDepth', channelDepth, 'BottomLoss', 0.5, ...
'NumPathsSource', 'Property', 'NumPaths', num_paths_inputs, 'PropagationSpeed', propSpeed, 'LossFrequencies', new_frequencies);

isopaths2 = phased.IsoSpeedUnderwaterPaths('ChannelDepth', channelDepth, 'BottomLoss', 0.5, ...
'NumPathsSource', 'Property', 'NumPaths', 1, 'PropagationSpeed', propSpeed, 'LossFrequencies', new_frequencies);

channel1 = phased.MultipathChannel('OperatingFrequency', OperatingFrequency); 
channel2 = phased.MultipathChannel('OperatingFrequency', OperatingFrequency); 
channel3 = phased.MultipathChannel('OperatingFrequency', OperatingFrequency); 
channel4 = phased.MultipathChannel('OperatingFrequency', OperatingFrequency); 

projector = phased.IsotropicProjector('VoltageResponse', 120); 
projRadiator = phased.Radiator('Sensor', projector, 'PropagationSpeed', propSpeed, 'OperatingFrequency', OperatingFrequency);

hydrophone = phased.IsotropicHydrophone('VoltageSensitivity', -150);
array = phased.ULA('Element', hydrophone, 'NumElements', numberofSensors, 'ElementSpacing', propSpeed/OperatingFrequency/2, 'ArrayAxis', 'y');
arrayCollector = phased.Collector('Sensor', array, 'PropagationSpeed', propSpeed, 'OperatingFrequency', OperatingFrequency);

Angles = -90:1:90; % Possible angles for signal arrival (degrees).
SNR = 10; % Single SNR value.
range = 3000; % Two range values.
sensor_depth = -80; % Single sensor depth.
fs = 12000; % Sampling frequency (Hz).
duration = 1; % Duration of the signal (seconds).
t = linspace(0, duration, fs ); % Time vector.
sound1 = cos(2*pi*1100*t)'; % Source signal (1 kHz cosine wave).

musicspatialspect = phased.MUSICEstimator('SensorArray',array,...
        'PropagationSpeed',propSpeed,'OperatingFrequency',...
        OperatingFrequency,'ScanAngles',-90:1:90,'DOAOutputPort',true,...
        'NumSignalsSource','Property','NumSignals',1);

Samples = 1; % Number of samples.
DOA_Labels = []; % Initialize DOA labels.
Input_matrix = []; % Initialize input matrix for training data.
Label_matrix = []; % Initialize label matrix for training data.

for pp = 1:length(sensor_depth) % This loop will run once
    for jj = 1:length(range) % This loop will run once
        for k = 1:length(SNR) % This loop will run once
            Y_training = zeros(Samples, fs, numberofSensors); % Initialize training data matrix.
            Y_label = zeros(Samples, fs, numberofSensors); % Initialize label data matrix.
            Binary_DOAs1 = zeros(Samples, length(Angles)); % Initialize binary DOA labels.
            
            for i = 1:Samples
                randomIndices = randperm(length(Angles), sources); % Randomly select source angles.
                randomThetas = Angles(randomIndices);
                binary_vector = zeros(size(Angles)); % Initialize binary vector for DOA.
                
                for ii = 1:length(randomThetas)
                    idx = find(Angles == randomThetas(ii));
                    binary_vector(idx) = 1;
                end
                Binary_DOAs1(i, :) = binary_vector;

                phi = 90; % Elevation angle.
                x1 = range(jj) * sind(phi) * cosd(randomThetas(1));  
                y1 = range(jj) * sind(phi) * sind(randomThetas(1));  
                z1 = range(jj) * cosd(phi);
                
                beaconPlatform1 = phased.Platform('InitialPosition', [x1; y1; sensor_depth + z1], 'Velocity', [0; 0; 0]);
                arrayPlatform1 = phased.Platform('InitialPosition', [0; 0; sensor_depth], 'Velocity', [0; 0; 0]);
                [pos_tx1, vel_tx1] = beaconPlatform1(duration); % Update positions of beacon.
                [pos_rx, vel_rx] = arrayPlatform1(duration); % Update positions of array.
                
               % isopaths = phased.IsoSpeedUnderwaterPaths('ChannelDepth', channelDepth, 'BottomLoss', 0.2,...
             %   'NumPathsSource', 'Property', 'NumPaths', num_paths_inputs, 'PropagationSpeed', propSpeed, 'LossFrequencies', new_frequencies);
                
                [paths1, dop1, aloss1, rcvang1, srcang1] = isopaths1(pos_tx1, pos_rx, vel_tx1, vel_rx, duration);  
                tsig1 = projRadiator(sound1, srcang1);
                rsig1 = channel1(tsig1, paths1, dop1, aloss1);
                recieved_signal1 = arrayCollector(rsig1, rcvang1);
                % combined_signal = recieved_signal1;
                % squared_magnitude = mean(abs(combined_signal).^2, 1);
                % signal_power = sum(squared_magnitude);
                % noise_power = signal_power / (10^(SNR / 10));
                % noise = zeros(numberofSensors, size(recieved_signal1, 1));
                % 
                % for iii = 1:size(recieved_signal1, 1)
                %     noise(:, iii) = sqrt(noise_power) / 2 .* (randn(numberofSensors, 1) + 1i * randn(numberofSensors, 1));
                % end
                % 
                % noise = noise';
             
                
                Y_training(i, :, :) = recieved_signal1;
                [Pseudo1,~] = musicspatialspect(recieved_signal1);
                plot(Angles,Pseudo1);
                figure;
                [paths2, dop2, aloss2, rcvang2, srcang2] = isopaths2(pos_tx1, pos_rx, vel_tx1, vel_rx, duration);  
                tsig2 = projRadiator(sound1, srcang2);
                rsig2 = channel3(tsig2, paths2, dop2, aloss2);
                recieved_signal2 = arrayCollector(rsig2, rcvang2);
                Y_label(i, :, :) = recieved_signal2;
                [Pseudo2,~] = musicspatialspect(recieved_signal2);
                plot(Angles,Pseudo2);
                


randomThetas
            end

            Input_matrix = [Input_matrix; Y_training];
            Label_matrix = [Label_matrix; Y_label];
            DOA_Labels = [DOA_Labels; Binary_DOAs1];
        end
    end
end






% first_sensor_signal = squeeze(Input_matrix(1, :, 1));% COL1
% spectrogram(real(first_sensor_signal), 512, 256, 512, fs, 'yaxis');
% title("Spectrogram of the col1 (Input Matrix) 5kHz");
% xlabel('Time (seconds)');
% ylabel('Frequency (kHz)');
% 
% figure;
% first_sensor_signal = squeeze(Label_matrix(1, :, 1));% COL1
% spectrogram(real(first_sensor_signal), 512, 256, 512, fs, 'yaxis');
% title("Spectrogram of the col1 (label Matrix) 5kHz");
% xlabel('Time (seconds)');
% ylabel('Frequency (kHz)');