from PIL import Image
import os

import os
import imageio

def create_mp4(image_folder, output_mp4, fps=30):
    images = []
    filenames = sorted(os.listdir(image_folder))
    
    for filename in filenames:
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Adjust file extensions as needed
            img_path = os.path.join(image_folder, filename)
            images.append(imageio.imread(img_path))

    imageio.mimsave(output_mp4, images, fps=fps)

# Usage example:
input_folder = "images"
output_mp4_path = "output.mp4"
fps = 24  # Adjust the frames per second (fps) as needed

create_mp4(input_folder, output_mp4_path, fps)