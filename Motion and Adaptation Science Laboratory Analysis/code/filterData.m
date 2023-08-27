function filtData = filterData(data, deg, fs, cutFreq,type)
    %passband = [cutFreq_low cutFreq_high]; % Passband frequency range in Hz
    passband = cutFreq;
    nyquist = fs/2;
    normalized_passband = passband / nyquist;
    [b,a] = butter(deg/2,normalized_passband,type);
    filtData = filtfilt(b,a,data);
end