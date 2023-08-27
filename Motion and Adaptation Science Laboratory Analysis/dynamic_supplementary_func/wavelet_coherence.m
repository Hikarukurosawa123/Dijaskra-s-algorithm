function [wcoh_all_mean_TA, wcoh_all_mean_MG, wcoh_all_mean_SOL, freq_wave] = wavelet_coherence(Cz_EEG, L_TA, L_MG, L_SOL)
    %wavelet coherence
    wcoh_all_TA = [];
    wcoh_all_MG = [];
    wcoh_all_SOL= [];
    
    for i = 1:size(Cz_EEG,3)
        [wcoh_TA,wcs,freq_wave] = wcoherence(Cz_EEG(1,:,i),L_TA(1,:,i), 250, "VoicesPerOctave", 20, "FrequencyLimits", [4 64], "NumScalesToSmooth", 18);
        [wcoh_MG,wcs,freq_wave] = wcoherence(Cz_EEG(1,:,i),L_MG(1,:,i), 250, "VoicesPerOctave", 20, "FrequencyLimits", [4 64], "NumScalesToSmooth", 18);
        [wcoh_SOL,wcs,freq_wave] = wcoherence(Cz_EEG(1,:,i),L_SOL(1,:,i), 250, "VoicesPerOctave", 20, "FrequencyLimits", [4 64], "NumScalesToSmooth", 18);
    
        wcoh_all_TA = cat(3, wcoh_all_TA, wcoh_TA);
        wcoh_all_MG = cat(3, wcoh_all_MG, wcoh_MG);
        wcoh_all_SOL = cat(3,wcoh_all_SOL, wcoh_SOL);
    end

    wcoh_all_mean_TA = mean(wcoh_all_TA, 3);
    wcoh_all_mean_MG = mean(wcoh_all_MG, 3);
    wcoh_all_mean_SOL = mean(wcoh_all_SOL,3);
end 

    