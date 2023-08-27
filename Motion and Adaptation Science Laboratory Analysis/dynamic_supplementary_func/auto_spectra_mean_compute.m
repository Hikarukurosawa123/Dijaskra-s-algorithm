function auto_spectra_mean =  auto_spectra_mean_compute(data)
    %description: computes the mean value of the auto-spectra

    %input: 
    % data - auto_spectra_all (n*m*l matrix: n = number of frequency bins, m = number of samples, l = number of epochs)
    %output n*m 
    auto_spectra_mean = mean(abs(data).^2,3);
end 
