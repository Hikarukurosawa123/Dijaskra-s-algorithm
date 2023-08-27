function finalData = processEmg(data)
    % convert to volts
    data = data/(2^16/2);
    % filter
    % rectify

    %subtract by the mean for each column (input must be column wise)
    data = data - mean(data,1);
    %passband = [20 500];
    %fs_EMG = 2000; 
    %data = filterData(data,4,fs_EMG,passband, "bandpass");

    finalData = abs(data);


    % Threshold calculation (do same with baseline, and detect using mean (after filter & rectification) + 3SD)
end