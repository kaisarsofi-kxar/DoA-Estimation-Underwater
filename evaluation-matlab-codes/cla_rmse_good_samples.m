clear all; clc;
rangedd={'1000'};

load(['./murtiza40_60/test_data_range/cleaned_AE/range/cleaned_range' rangedd{1} '.mat'])


% Constants
propSpeed = 1520; % Speed of sound in water (m/s).
OperatingFrequency = 4000; % Operating frequency (Hz).
numberofSensors = 12; % Number of sensors in the array.
Angles = -90:1:90-1; % Possible angles for signal arrival (degrees).
actual_angles = sort([20, 40]); % Actual angles (degrees)
sources=2;

num_samples = 300;
good_samples = 0;
% Array setup
hydrophone = phased.IsotropicHydrophone('VoltageSensitivity', -150);
array = phased.ULA('Element', hydrophone, 'NumElements', numberofSensors, ...
                   'ElementSpacing', propSpeed/OperatingFrequency/2, 'ArrayAxis', 'y');

% MUSIC estimator setup
musicspatialspect = phased.MUSICEstimator('SensorArray', array, ...
    'PropagationSpeed', propSpeed, 'OperatingFrequency', OperatingFrequency, ...
    'ScanAngles', Angles, 'DOAOutputPort', true, 'NumSignalsSource', 'Property', 'NumSignals', 2);

% Initialize matrix to store differences
difference_matrix = zeros(num_samples, 2);

for sample_index = 1:num_samples
    % Extract the signal for the current sample
    selected_signal = cleaned_input(sample_index, :, :);

    % Apply MUSIC algorithm to the selected signal
    received_signal = squeeze(selected_signal);
    [PseudoSpectrum, doa_source] = musicspatialspect(received_signal);
    plot(Angles,PseudoSpectrum./ max(abs(PseudoSpectrum)))
    hold on

    normalized_spectrum = PseudoSpectrum/ max(abs(PseudoSpectrum));
    % Find peaks without MinPeakHeight
    [peaks, locs] = findpeaks(normalized_spectrum, Angles, 'MinPeakHeight',0.3);
    locs =sort(locs);
        

    % Ensure we have at least 2 peaks to compare
    if length(locs) >= 2
        % Calculate the differences
        difference_matrix(sample_index, 1) = locs(1) - actual_angles(1);
        difference_matrix(sample_index, 2) = locs(2) - actual_angles(2);
        good_samples= good_samples + 1;
        disp(locs)

    else
        diffrence_sum = NaN;
    end
end




% function rmse = special_case()
% 
%     for sample_index =1:num_samples
% 
%         selected_signal = cleaned_input(sample_index, :, :);
%         % Apply MUSIC algorithm to the selected signal
%         received_signal = squeeze(selected_signal);
%         [PseudoSpectrum, doa_source] = musicspatialspect(received_signal);
%          normalized_spectrum = PseudoSpectrum/ max(abs(PseudoSpectrum));
%         % Find peaks without MinPeakHeight
%         [peaks, locs] = findpeaks(normalized_spectrum, Angles, 'MinPeakHeight',0.3);
% 
%         locs =sort(locs);
%         disp(locs)
% 
%         if length(locs)>=2
%             difference_matrix(sample_index, 1) = locs(1) - actual_angles(1);
%             difference_matrix(sample_index, 2) = locs(2) - actual_angles(2);
% 
%         end
% 
% 
% 
% 
% 
% 
% 
% 
%     end
% end

square_matrix = difference_matrix.^2;
sum_square_matrix = sum(sum(square_matrix));
rmse= sqrt(sum_square_matrix/(sources*good_samples));

disp("Good_samples: ")
disp(good_samples)
disp("RMSE: ")
disp(rmse)

