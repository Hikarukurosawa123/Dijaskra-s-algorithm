function [cross_spectra_all] =  cross_spectra_compute(data1, data2)
    %description: computes the cross spectra of the auto spectra of two
    %signals 

    %input: 
    %data1 (n*m*l complex number spectrogram matrix: n = number of frequency bins, m = number of samples, l = number of epochs)
    %data2 (n*m*l complex number spectrogram matrix: n = number of frequency bins, m = number of samples, l = number of epochs)
    %output: n*m matrix of the cross_spectra
    cross_spectra_all = data1 .* conj(data2);
end 
