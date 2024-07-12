clear all; close all; clc;

% Load the .mat file
load('cleaned_matrix_45_samples_simple_AE.mat'); 
load('test_covariance_new_data3.mat');

% Extract a particular signal from the loaded data
selected_signal_index = 1; 
selected_signal_cleaned = data(selected_signal_index, :, :);
selected_signal_original = Input_matrix(selected_signal_index, :, :);

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
received_signal_cleaned = squeeze(selected_signal_cleaned);
received_signal_original = squeeze(selected_signal_original);
[PseudoSpectrum_cleaned, ~] = musicspatialspect(received_signal_cleaned);
[PseudoSpectrum_original, ~] = musicspatialspect(received_signal_original);
% Plot the MUSIC pseudo-spectrum
figure;
plot(Angles, PseudoSpectrum_cleaned);
title('MUSIC Pseudo-Spectrum');
xlabel('Angle (degrees)');
ylabel('Pseudo Spectrum cleaned');
grid on;


figure;
plot(Angles, PseudoSpectrum_original);
title('MUSIC Pseudo-Spectrum');
xlabel('Angle (degrees)');
ylabel('Pseudo Spectrum original');
grid on;
% % Display the estimated DOAs
% disp('Estimated DOAs:');
% disp(DOAs);
