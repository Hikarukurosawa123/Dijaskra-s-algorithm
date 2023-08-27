function coherence = surrogate_coherence(Cz, EMG)
%computes the surrogate coherence by pairing the ith cycle of EEG with jth
%cycle of EMG and observe whether the CMC values are due to coincidence or
%actual synchronicity 

%Cz & EMG is a matrix quantity in time series (each collumn corresponds to each
%epoch
    window_length = 48;
    window = hanning(window_length);
    fs = 250; %sampling frequency 
    overlap_gap = 6;
    Cz_auto_spectra_all = [];
    EMG_auto_spectra_all = [];
    for i = 1:size(Cz, 2)
        for j = 1:size(EMG,2)
            if i ~= j
                Cz_trial = Cz(:,i);
                EMG_trial = EMG(:,j);
                [Cz_auto_spectra,f] = stft(Cz_trial, fs, Window = window, OverlapLength = window_length - overlap_gap,FFTLength = 64, FrequencyRange = "onesided");%centered 
                [EMG_auto_spectra,f] = stft(EMG_trial, fs, Window = window, OverlapLength = window_length - overlap_gap,FFTLength = 64, FrequencyRange = "onesided");%centered 
              
                Cz_auto_spectra_all = cat(3, Cz_auto_spectra_all, Cz_auto_spectra);
                EMG_auto_spectra_all = cat(3, EMG_auto_spectra_all, EMG_auto_spectra);

            end
        end 
    end 
    
    Cz_auto_spectra_all_mean = mean(abs(Cz_auto_spectra_all).^2, 3);
    EMG_auto_spectra_all_mean = mean(abs(EMG_auto_spectra_all).^2, 3);
    Cz_EMG_cross_spectra_all_mean = mean(Cz_auto_spectra_all.*conj(EMG_auto_spectra_all),3);
    coherence = abs(Cz_EMG_cross_spectra_all_mean).^2 ./ (Cz_auto_spectra_all_mean.*EMG_auto_spectra_all_mean);
end 
