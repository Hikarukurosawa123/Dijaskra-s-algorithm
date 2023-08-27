function [lift_dist lift_dist_mean] = get_step_length(mocap_data, lift_mocap)
    %description: find step length - find maximum and minimum coordinates right before and after the foot off event
    lift_min = zeros(1,length(lift_mocap));
    lift_max = zeros(1,length(lift_mocap));
    search_window = 200; %1 second search window 
    
    for i = 1:length(lift_mocap)
        lift_mocap_down = floor(lift_mocap(i)/10);
        lift_min(1,i) = min(mocap_data(lift_mocap_down - search_window: lift_mocap_down));
        lift_max(1,i) = max(mocap_data(lift_mocap_down: lift_mocap_down + search_window));
    end 
    
    lift_dist = lift_max - lift_min;
    lift_dist_mean = mean(lift_dist);

end 