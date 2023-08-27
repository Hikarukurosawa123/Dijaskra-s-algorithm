function [IAC_mean] = IAC_compute(data1, data2)

    %description: computes the mean of the instantaneous amplitude correlation (IAC) of two signals 

    %input: 
    % data1 (n*m*l complex number spectrogram matrix: n = number of frequency bins, m = number of samples, l = number of epochs)
    % data2 (n*m*l complex number spectrogram matrix: n = number of frequency bins, m = number of samples, l = number of epochs)
    % output (n*m matrix: n = number of frequency bins, m = number of samples)
    IAC_trial_data1 = [];
    IAC_trial_data2 = [];
    for i = 1:size(data1,3)
    
        amplitude_data1 = abs(hilbert(abs(transpose(data1(:,:,i)))).^2);
        amplitude_data2 = abs(hilbert(abs(transpose(data2(:,:,i)))).^2);
      
        amplitude_data1_norm = normalize(transpose(amplitude_data1(:,:)),2);
        amplitude_data2_norm = normalize(transpose(amplitude_data2(:,:)),2);

        IAC_trial_data1 = cat(3, IAC_trial_data1, amplitude_data1_norm);
        IAC_trial_data2 = cat(3, IAC_trial_data2, amplitude_data2_norm);

        IAC_trial_all = IAC_trial_data1.*IAC_trial_data2;
    end 
    IAC_mean = mean(IAC_trial_all, 3);
end 

