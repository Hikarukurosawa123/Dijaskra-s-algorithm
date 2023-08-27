function [IAC_trial_all_Cz_TA, IAC_trial_all_Cz_SOL, IAC_trial_all_Cz_MG, input_Cz_all, input_MG_all, input_TA_all, input_SOL_all] = IAC_computation(Cz_auto_spectra_all, L_MG_auto_spectra_all, L_TA_auto_spectra_all, L_SOL_auto_spectra_all)
    IAC_trial_all_Cz_MG= [];
    IAC_trial_all_Cz_TA = [];
    IAC_trial_all_Cz_SOL = [];
    input_Cz_all = [];
    input_MG_all = [];
    input_TA_all = [];
    input_SOL_all = [];

    for i = 1:size(Cz_auto_spectra_all,3)
    
        input_Cz = abs(hilbert(abs(transpose(Cz_auto_spectra_all(:,:,i)))).^2);
        input_MG = abs(hilbert(abs(transpose(L_MG_auto_spectra_all(:,:,i)))).^2);
        input_TA = abs(hilbert(abs(transpose(L_TA_auto_spectra_all(:,:,i)))).^2);
        input_SOL = abs(hilbert(abs(transpose(L_SOL_auto_spectra_all(:,:,i)))).^2);

        
        input_Cz_norm = normalize(input_Cz(:,:)',2);
        input_TA_norm = normalize(input_TA(:,:)',2);
        input_MG_norm = normalize(input_MG(:,:)',2);
        input_SOL_norm = normalize(input_SOL(:,:)',2);

        input_Cz_all = cat(3, input_Cz_all, input_Cz_norm);
        input_MG_all = cat(3, input_MG_all, input_TA_norm);
        input_TA_all = cat(3, input_TA_all, input_MG_norm);
        input_SOL_all = cat(3, input_SOL_all, input_SOL_norm);

        IAC_trial_Cz_MG = input_Cz_norm.*input_MG_norm;
        IAC_trial_Cz_TA = input_Cz_norm.*input_TA_norm;
        IAC_trial_Cz_SOL = input_Cz_norm.*input_SOL_norm;
    
        IAC_trial_all_Cz_MG = cat(3,IAC_trial_all_Cz_MG, IAC_trial_Cz_MG);
        IAC_trial_all_Cz_TA = cat(3,IAC_trial_all_Cz_TA, IAC_trial_Cz_TA);
        IAC_trial_all_Cz_SOL = cat(3,IAC_trial_all_Cz_SOL, IAC_trial_Cz_SOL);
    
    end 
end 

