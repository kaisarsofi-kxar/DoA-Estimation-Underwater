clear all; clc;

% Load the .mat file


% base_filename = 'cleaned_data_range';
% range_value=4000;
% filename = sprintf('./resnet34_covariance/trained_model_with_dpr_v_adptive_lr/cleaned_data_variable_range/cleaned_mat_range%d/%s%d.mat',range_value,base_filename,range_value);
% data =load(filename); 
% % load('./resnet34_covariance/trained_model_with_dpr_v_adptive_lr/cleaned_data/cleaned_mat_snr_20/cleaned_data_SNR_20.mat'); 
% variable_name = sprintf('cleaned_input_range%d',range_value);
% if isfield(data, variable_name)
%     cleaned_input = data.(variable_name);
% end

load('./murtiza40_60/cleaned_simple_AE/depth/depth_snr_5/cleaned_simple_AE_depth_30.mat');
selected_signal_index=76;
selected_signal = cleaned_input(selected_signal_index, :, :);

% Constants
propSpeed = 1520; % Speed of sound in water (m/s).
OperatingFrequency = 4000; % Operating frequency (Hz).
numberofSensors = 12; % Number of sensors in the array.
Angles = -90:1:90; % Possible angles for signal arrival (degrees).

% Array setup
hydrophone = phased.IsotropicHydrophone('VoltageSensitivity', -150);
array = phased.ULA('Element', hydrophone, 'NumElements', numberofSensors, ...
                   'ElementSpacing', propSpeed/OperatingFrequency/2, 'ArrayAxis', 'y');


% MUSIC estimator setup
musicspatialspect = phased.MUSICEstimator('SensorArray', array, ...
    'PropagationSpeed', propSpeed, 'OperatingFrequency', OperatingFrequency, ...
    'ScanAngles', Angles, 'DOAOutputPort', true, 'NumSignalsSource', 'Property', 'NumSignals', 2);

% Apply MUSIC algorithm to the selected signal
received_signal = squeeze(selected_signal);
[PseudoSpectrum, ~] = musicspatialspect(received_signal);
% Plot the MUSIC pseudo-spectrum


figure;
plot(Angles, PseudoSpectrum/max(PseudoSpectrum));
title('MUSIC Pseudo-Spectrum');
xlabel('Angle (degrees)');
ylabel('Pseudo Spectrum');
grid on;

nn = PseudoSpectrum/max(PseudoSpectrum);

% % Display the estimated DOAs
% disp('Estimated DOAs:');
% disp(DOAs);
% Find peaks with custom parameters
[peaks, locs] = findpeaks(nn, Angles, 'MinPeakHeight', .2);

disp(locs)

% Plot the signal and highlight the peaks
% Plot the pseudo-spectrum
% figure;
% plot(Angles, nn);
% hold on;
% 
% % Highlight the peaks
% plot(locs, peaks, 'r^', 'MarkerFaceColor', 'r');  % Mark peaks with red triangles
% title('MUSIC Pseudo-Spectrum with Peaks');
% xlabel('Angle (degrees)');
% ylabel('Pseudo Spectrum');
% legend('Pseudo Spectrum', 'Peaks');
% hold off;