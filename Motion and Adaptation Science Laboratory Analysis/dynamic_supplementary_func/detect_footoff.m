function [closest_indexes_lift closest_indexes_stamp] = detect_footoff(lift_mocap, lift_F1Z, stamp_F3Z, num_steps)
    %description: detect the foot off and foot contact event right before and after the stepping event detected by the mocap data, respectively

    %input: 
    %lift_mocap: n*1 vector representing the time series of the motion capture data in binary value (0 - pre stepping, 1 - post stepping)
    %lift_F1Z: n*1 vector representing the indexes of the points at which the vertical comopnent of the front force plate went from high -> low  
    %lift_F3Z: n*1 vector representing the indexes of the points at which the vertical comopnent of the back force plate went from low -> high  

    closest_indexes_lift = [];
    closest_indexes_stamp = [];
    
    for i = 1:num_steps %set the number of segments to use (either length(lift_mocap) or set it manually if signal is gone)
        value = lift_mocap(i);
        lift_F1Z_less = lift_F1Z(lift_F1Z < value);
        stamp_F3Z_greater= stamp_F3Z(stamp_F3Z > value);
    
        if isempty(lift_F1Z_less) == 0
            closest_value = lift_F1Z(length(lift_F1Z_less));
            closest_indexes_lift = [closest_indexes_lift closest_value];
        end 
        if isempty(stamp_F3Z_greater) == 0
            closest_value_F3Z = stamp_F3Z_greater(1);
            closest_indexes_stamp = [closest_indexes_stamp closest_value_F3Z];
        end 
    end 
end 