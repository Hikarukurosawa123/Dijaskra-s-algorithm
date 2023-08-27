function [mask_matrix_TA, mask_matrix_MG, mask_matrix_SOL, phase_mean_Cz_TA, phase_mean_Cz_MG, phase_mean_Cz_SOL] = mask_matrix(coherence_TA, coherence_MG, coherence_SOL, phase_mean_Cz_TA, phase_mean_Cz_MG, phase_mean_Cz_SOL, significance_threshold)
%mask the values based on the points at which CMC value is below
%significant threshold
    mask_matrix_TA = ones(size(coherence_TA));
    [x_TA, y_TA, val] = find(coherence_TA < significance_threshold);
    for i = 1:length(x_TA)
        phase_mean_Cz_TA(x_TA(i), y_TA(i)) = NaN;
        mask_matrix_TA(x_TA(i), y_TA(i)) = NaN;
    end 
    
    mask_matrix_MG = ones(size(coherence_MG));
    [x_MG, y_MG, val] = find(coherence_MG < significance_threshold);
    for i = 1:length(x_MG)
        phase_mean_Cz_MG(x_MG(i), y_MG(i)) = NaN;
        mask_matrix_MG(x_MG(i), y_MG(i)) = NaN;
    end 
    
    mask_matrix_SOL = ones(size(coherence_SOL));
    [x_SOL, y_SOL, val] = find(coherence_SOL < significance_threshold);
    for i = 1:length(x_SOL)
        phase_mean_Cz_SOL(x_SOL(i), y_SOL(i)) = NaN;
        mask_matrix_SOL(x_SOL(i), y_SOL(i)) = NaN;
    end 
end 