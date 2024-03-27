% Define paths for input and output directories
input_directory = '../Channel Decomposed/';
output_noised_directory = '../Noised/';
output_denoised_directory = '../Denoised/';

% Get a list of all files in the input directory
files = dir(fullfile(input_directory, '*.JPG'));

% Iterate over each file
for i = 1:length(files)
    % Read the image
    image = imread(fullfile(input_directory, files(i).name));
    
    % Resize the image
    image_resized = imresize(image, [512, 512]);
    
    % Add noise
    noised_image = imnoise(image_resized, 'speckle');
    
    % Save the noised image
    imwrite(noised_image, fullfile(output_noised_directory, ['noised_', files(i).name]));
    
    % Convert to double
    noised_image = double(noised_image);
    
    % Run denoising algorithm
    denoised_image = denoising_dwt(noised_image);
    
    % Convert to uint8
    denoised_image = uint8(denoised_image);
    
    % Save the denoised image
    imwrite(denoised_image, fullfile(output_denoised_directory, ['denoised_', files(i).name]));
end
