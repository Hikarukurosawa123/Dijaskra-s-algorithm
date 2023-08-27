function data_segments = remove_trials_func(data_segments, remove_trials)
    %description: remove epochs defined by the index number specified by
    %"remove_trials" matrix

    %input: 
    % data_segments (n*m*l matrix: n = number of channels, m = number of samples, l = number of epochs)
    % remove_trials (n*1 vector: n = number of epochs to remove)
    %output n*m*(l-length(remove_trials))
    [~, ~, num_trials] = size(data_segments);
    keep_indices = setdiff(1:num_trials, remove_trials);
    data_segments = data_segments(:, :, keep_indices);
end 