%% Open image
image = imread('../Channel Decomposed/TRAIN000009G.JPG')

% figure('name', 'Original Image')
% imshow(image, [0, 255])
    
[rows, columns, numberOfColorChannels] = size(image)
disp(class(image))

%% resize 512
image_resized = imresize(image, [512, 512])
image_resized_double = double(image_resized)
% imshow(image_resized, [0, 255])

%% Add noise
noised_image = imnoise(image_resized,'speckle')
noised_image = double(noised_image)
figure('name','noised')
imshow(noised_image, [0, 255])

%% Run local adaptive image denoising algorithm using dual-tree DWT.
denoised_image = denoising_dwt(noised_image);
figure('name','denoised')
imshow(denoised_image, [0, 255])

%% Calculate the error
err = image_resized_double - denoised_image;

%% Calculate the PSNR value
PSNR = 20*log10(256/std(err(:)))