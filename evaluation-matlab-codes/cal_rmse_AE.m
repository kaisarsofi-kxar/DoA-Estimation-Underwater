clear all; clc;


% Load the .mat file from best_resnet_dproout_adaptive_lr
base_filename = 'cleaned_data_depth_';
range_value = 110;
filename = sprintf('./simple_AE_with_dropouts/trained_model_v_snsr_data/cleaned_data_variable_depth/cleaned_mat_depth_%d/%s%d.mat',range_value,base_filename,range_value);
data = load(filename); 

variable_name = sprintf('cleaned_input_depth_%d',range_value);
if isfield(data, variable_name)
    cleaned_input = data.(variable_name);

end

% Constants
propSpeed = 1520; % Speed of sound in water (m/s).
OperatingFrequency = 4000; % Operating frequency (Hz).
numberofSensors = 12; % Number of sensors in the array.
Angles = -90:1:90-1; % Possible angles for signal arrival (degrees).
actual_angles = [20, 40]; % Actual angles (degrees)
sources=2;

num_samples = 250;

% Array setup
hydrophone = phased.IsotropicHydrophone('VoltageSensitivity', -150);
array = phased.ULA('Element', hydrophone, 'NumElements', numberofSensors, ...
                   'ElementSpacing', propSpeed/OperatingFrequency/2, 'ArrayAxis', 'y');

% MUSIC estimator setup
musicspatialspect = phased.MUSICEstimator('SensorArray', array, ...
    'PropagationSpeed', propSpeed, 'OperatingFrequency', OperatingFrequency, ...
    'ScanAngles', Angles, 'DOAOutputPort', true, 'NumSignalsSource', 'Property', 'NumSignals', 2);

% Initialize matrix to store differences
difference_matrix = zeros(200, 2);

for sample_index = 1:num_samples
    % Extract the signal for the current sample
    selected_signal = cleaned_input(sample_index, :, :);

    % Apply MUSIC algorithm to the selected signal
    received_signal = squeeze(selected_signal);
    [PseudoSpectrum, ~] = musicspatialspect(received_signal);
    

    normalized_spectrum = PseudoSpectrum/ max(PseudoSpectrum);
    % Find peaks without MinPeakHeight
    [peaks, locs] = findpeaks(normalized_spectrum, Angles, 'MinPeakHeight',0.3);
    disp(locs)
    

    % Ensure we have at least 2 peaks to compare
    if length(locs) >= 2
        % Calculate the differences
        difference_matrix(sample_index, 1) = locs(1) - actual_angles(1);
        difference_matrix(sample_index, 2) = locs(2) - actual_angles(2);
    else
        difference_sum = NaN;
    end
end

square_matrix = difference_matrix.^2;
sum_square_matrix = sum(sum(square_matrix));
rmse= sqrt(sum_square_matrix/(sources*num_samples));
disp("RMSE: ")
disp(rmse)
