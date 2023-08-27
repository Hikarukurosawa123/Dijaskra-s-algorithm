function release_indexes = find_transducer_release(force, threshold, closest_indexes_lift, search_window_length)
    %description: finds the timepoint of the cable release timing in the
    %lean and release test

    %input: 
    %force - n*1 vector of force plate data 
    %threshold - scalar value of the threshold, cable release is defined as
    %the first point the difference of two consecutive values exceeded the threshold value 
    %closest_index_lift - n*1 vector representing the indexes at which the
    %foot off occured
    %search_window_length - scalar value to set the window length of
    %searching the cable release timing with respect to the foot off index
    release_indexes = zeros(1,length(closest_indexes_lift));
    for i = 1:length(closest_indexes_lift)
        end_index = closest_indexes_lift(1,i);
        beg_index = end_index - search_window_length;
        transitions = find(diff(force(beg_index:end_index)) >= threshold);
        release_indexes(1,i) = transitions(1) + beg_index;
    end
end 