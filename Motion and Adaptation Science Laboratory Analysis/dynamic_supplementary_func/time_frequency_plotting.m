function time_frequency_plotting(time, f, data, title_name, save_bool, file_name, folder_path)
    if ~exist('save_bool','var')
      save_bool = 0;
      file_name = [];
      folder_path = [];
    end 
    figure; 
    imagesc(time, f, data);
    colormap('jet'); % Choose a colormap of your preference
    xlabel('Time'); % Label for the x-axis
    ylabel('Frequency'); % Label for the y-axis
    colorbar; % Add a colorbar with label
    set(gca,'YDir','normal')
    ylim([4 50])
    title(title_name)
    if save_bool == 1
        file_path = fullfile(folder_path, file_name);
        saveas(gcf, file_path, 'png')
    end 
    hold on;
end 
