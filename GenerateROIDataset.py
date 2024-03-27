import time
import cv2
import os
import multiprocessing
import numpy as np
import pandas as pd

READ_FOLDER = 'JustRAIGS_Train'
SAVE_FOLDER = 'ROI'

def GetROI (filename):
    # Start time
    start_time = time.time()

    # Open the image
    image = cv2.imread(f'{READ_FOLDER}/{filename}')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform top-hat and bottom-hat transform
    kernel_size = (100,100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
    bottomhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)

    # Combine the original image with the top-hat and bottom-hat transforms
    enhanced_image = cv2.add(gray_image, tophat)
    enhanced_image = cv2.subtract(enhanced_image, bottomhat)

    # Flatten the image array to 1D
    flattened_image = enhanced_image.flatten()

    # Sort the flattened array in descending order
    sorted_values = np.sort(flattened_image)[::-1]

    # Determine the threshold value
    threshold_index = int(len(sorted_values) * 0.065)
    threshold_value = sorted_values[threshold_index]

    # Create a binary mask of pixels above the threshold
    mask = (enhanced_image >= threshold_value).astype(np.uint8)

    # Mask the original image to retain only the top 6.5% brightest pixels
    brightest_region = enhanced_image * mask

    # Extract the green channel
    green_channel = image[:, :, 1]

    # Apply CLAHE on the green channel
    # clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    clahe = cv2.createCLAHE()
    clahe_green = clahe.apply(green_channel)

    # Perform bottom-hat transform on the green channel
    nerve_tophat = cv2.morphologyEx(clahe_green, cv2.MORPH_TOPHAT, kernel)
    nerve_bottomhat = cv2.morphologyEx(clahe_green, cv2.MORPH_BLACKHAT, kernel)

    # segmented_nerve_image = cv2.subtract(nerve_tophat, nerve_bottomhat)
    nerve_image = cv2.subtract(nerve_bottomhat, nerve_tophat)

    # Combine segmented nerve with enhanced
    combined_image = cv2.add(brightest_region, nerve_image)

    # Define window size for sliding window
    window_size = 250
    window = (window_size, window_size)
    stride = 15

    # Calculate average intensity for each image
    avg_intensity_brightest_region = sliding_window_avg(brightest_region, window, stride)
    avg_intensity_segmented = sliding_window_avg(nerve_image, window, stride)
    avg_intensity_combined = sliding_window_avg(combined_image, window, stride)

    # Normalize each result
    enhanced_normalized = (avg_intensity_brightest_region - np.min(avg_intensity_brightest_region)) / (np.max(avg_intensity_brightest_region) - np.min(avg_intensity_brightest_region))
    segmented_normalized = (avg_intensity_segmented - np.min(avg_intensity_segmented)) / (np.max(avg_intensity_segmented) - np.min(avg_intensity_segmented))
    combined_normalized = (avg_intensity_combined - np.min(avg_intensity_combined)) / (np.max(avg_intensity_combined) - np.min(avg_intensity_combined))

    # Find brightest spot 
    summed_of_all = enhanced_normalized + segmented_normalized + combined_normalized
    avg_of_all = summed_of_all / 3

    # Find the index of the highest value
    max_index = np.argmax(avg_of_all)

    # Convert the flattened index to 2D index
    max_index_2d = np.unravel_index(max_index, avg_of_all.shape)

    # Specify the center coordinates
    center_y, center_x = max_index_2d

    crop_size = 3 * window_size
    right_bias = 0
    bottom_bias = 0

    # Calculate the top-left corner coordinates of the crop region
    top_left_x = max(0, center_x - crop_size//2 + right_bias)
    top_left_y = max(0, center_y - crop_size//2 + bottom_bias)

    # Calculate the bottom-right corner coordinates of the crop region
    bottom_right_x = min(image.shape[1], center_x + crop_size//2 + right_bias)
    bottom_right_y = min(image.shape[0], center_y + crop_size//2 + bottom_bias)

    # Crop the image
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Resize the cropped image to 800x800 if necessary
    cropped_image = cv2.resize(cropped_image, (crop_size, crop_size))
    cv2.imwrite(f'{SAVE_FOLDER}/{filename}', cropped_image)

    # End time
    end_time = time.time()

    # Calculate runtime
    runtime = end_time - start_time
    print(f'{filename} processed for {runtime} seconds')


# Function to perform sliding window and calculate average intensity
def sliding_window_avg(image, window_size, stride):
    height, width = image.shape
    padding_y = window_size[0] // 2
    padding_x = window_size[1] // 2
    
    end_y = height - padding_y 
    end_x = width - padding_x
    result = np.empty(image.shape)

    # Iterate through the image with the sliding window
    for y in range(padding_y, end_y, stride):
        for x in range(padding_x, end_x, stride):
            # Calculate the coordinates of the center of the window
            center_y = y + padding_y
            center_x = x + padding_x

            # Adjust the window boundaries based on the center coordinates
            window_y_start = max(0, center_y - padding_y)
            window_y_end = min(height, center_y + padding_y)
            window_x_start = max(0, center_x - padding_x)
            window_x_end = min(width, center_x + padding_x)

            window = image[window_y_start:window_y_end, window_x_start:window_x_end]
            result[center_y, center_x] = np.mean(window)

    return result  

# Function to process each split array
def process_files(split_files):
    for filename in split_files:
        GetROI(filename)

if __name__ == "__main__":

    GetROI('TRAIN095523.JPG')
    exit()

    # Number of processes
    num_processes = 16 

    # Get filenames in each directory
    # filenames1 = set(os.listdir(READ_FOLDER))
    filenames2 = set(os.listdir(SAVE_FOLDER))
    

    # Specify the path to your CSV file
    csv_file_path = 'JustRAIGS_Train_labels.csv'
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, delimiter=';')
    df['Eye ID'] = df['Eye ID'] + '.JPG'
    filenames1 = set(df[df['Final Label'] == 'RG']['Eye ID'])

    # Get filenames that exist in directory1 but not in directory2
    filenames = filenames1 - filenames2
    # filenames = filenames1

    # Split filenames into N chunks
    chunk_size = len(filenames) // num_processes
    filename_chunks = [list(filenames)[i:i+chunk_size] for i in range(0, len(filenames), chunk_size)]

    # Create and start a multiprocessing process for each chunk
    processes = []
    for chunk in filename_chunks:
        process = multiprocessing.Process(target=process_files, args=(chunk,))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()