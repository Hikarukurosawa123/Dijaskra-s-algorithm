function discretized_data = discretize(data, threshold)
    %description: discretize the values into 0 and 1 based on the values
    %being below or above the threshold 

    %input: n*1 vector (n = number of samples)
    %output: n*1 vector (n = number of samples)
    data_low = find(data < threshold); 
    data_high = find(data >= threshold); 
    data(data_low) = 0;
    data(data_high) = 1;
    discretized_data = data;
end 