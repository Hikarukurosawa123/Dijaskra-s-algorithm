function [f, spectrogram_all] =  auto_spectra_compute(data, fs, window_length, overlap_gap)

    %description: computes the spectrogram of a dataset

    %input: 
    % data (n*m*l real number matrix: n = number of channels, m = number of samples, l = number of epochs)
    % output (n*m*l matrix: n = number of frequency bins, m = number of samples, l = number of epochs)
    window = hann(window_length);  
    spectrogram_all = [];

    for i = 1:size(data,3)
        [spectrogram,f] = stft(data(1,:,i), fs, Window = window, OverlapLength = window_length - overlap_gap,FFTLength = 64, FrequencyRange = "onesided");%centered 
        spectrogram_all = cat(3, spectrogram_all, spectrogram);
    end 
end 
