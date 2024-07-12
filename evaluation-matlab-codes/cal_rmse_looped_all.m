clear all; clc;

% Constants
propSpeed = 1520; % Speed of sound in water (m/s).
OperatingFrequency = 4000; % Operating frequency (Hz).
numberofSensors = 12; % Number of sensors in the array.
Angles = -90:1:90-1; % Possible angles for signal arrival (degrees).
actual_angles = sort([40, 60]); % Actual angles (degrees)
sources=2;

% Array setup
hydrophone = phased.IsotropicHydrophone('VoltageSensitivity', -150);
array = phased.ULA('Element', hydrophone, 'NumElements', numberofSensors, ...
                   'ElementSpacing', propSpeed/OperatingFrequency/2, 'ArrayAxis', 'y');

% MUSIC estimator setup
musicspatialspect = phased.MUSICEstimator('SensorArray', array, ...
    'PropagationSpeed', propSpeed, 'OperatingFrequency', OperatingFrequency, ...
    'ScanAngles', Angles, 'DOAOutputPort', true, 'NumSignalsSource', 'Property', 'NumSignals', 2);






% SNR_values = {'20','15','10','5','0','_5','_10'};


SNR_values = {'_5','0','5'};

for ssr = 1:length(SNR_values)

    % range_val = {'1000','3000','5000','7000'};
    depth = {'10','30','50','70','90','110'};
    % numpaths={'1','11','21','31','41','51'};
    fprintf('rmse simple AE numpaths snr %s \n',SNR_values{ssr});

    for val = 1:length(depth)

        load(['./Murtiza40_60/cleaned_res34//depth/depth_snr' SNR_values{ssr} '/cleaned_res34_depth_' depth{val} '.mat'])
       
        num_samples = 1000;
        good_samples = 0;
       
        % Initialize matrix to store differences
        difference_matrix = zeros(num_samples, 2);
        
        for sample_index = 1:num_samples
            % Extract the signal for the current sample
            selected_signal = cleaned_input(sample_index, :, :);
        
            % Apply MUSIC algorithm to the selected signal
            received_signal = squeeze(selected_signal);
            [PseudoSpectrum, doa_source] = musicspatialspect(received_signal);
            % plot(Angles,PseudoSpectrum./ max(abs(PseudoSpectrum)))
            % hold on
        
            normalized_spectrum = PseudoSpectrum/ max(abs(PseudoSpectrum));
            % Find peaks without MinPeakHeight
            [peaks, locs] = findpeaks(normalized_spectrum, Angles, 'MinPeakHeight',0.2);
            locs = sort(locs);
                
        
            % Ensure we have at least 2 peaks to compare
            if length(locs) >= 2
                % Calculate the differences
                difference_matrix(sample_index, 1) = locs(1) - actual_angles(1);
                difference_matrix(sample_index, 2) = locs(2) - actual_angles(2);
                good_samples= good_samples + 1;
            else
                diffrence_sum = NaN;
            end
        end
        square_matrix = difference_matrix.^2;
        sum_square_matrix = sum(sum(square_matrix));
        rmse= sqrt(sum_square_matrix/(sources*good_samples));
        % disp("Good_samples: ")
        % disp(good_samples)
        fprintf('%.4f\n',rmse)
    end
end
