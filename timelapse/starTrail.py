import cv2
import numpy as np
import os

# Directory containing all-sky images
image_directory = 'path_to_your_images_directory'

# List all image files in the directory
image_files = sorted([os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('.jpg', '.png', '.jpeg'))])

# Read the first image to get the dimensions
first_image = cv2.imread(image_files[0])
height, width, channels = first_image.shape

# Initialize an empty array for the stacked image
stacked_image = np.zeros((height, width, channels), dtype=np.uint8)

# Stack all images
for image_file in image_files:
    image = cv2.imread(image_file)
    # Use maximum value to keep the star trails bright
    stacked_image = np.maximum(stacked_image, image)

# Save the final stacked image
output_filename = 'startrails_output.png'
cv2.imwrite(output_filename, stacked_image)

print(f'Star trail image saved as {output_filename}')
