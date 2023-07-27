import os
import imageio
import numpy as np
import concurrent.futures
from PIL import Image
from tqdm import tqdm

def resize_with_aspect_ratio(img, scale):
    width, height = img.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    return img.resize((new_width, new_height), Image.LANCZOS)

def append_vertical_frame(args):
    img_path1, img_path2, resize_scale = args
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)

    # Resize images if necessary
    if resize_scale != 1.0:
        img1 = resize_with_aspect_ratio(img1, resize_scale)
        img2 = resize_with_aspect_ratio(img2, resize_scale)

    combined_frame = np.concatenate((img1, img2), axis=0)
    return combined_frame

def create_vertical_gif(folder1_images, folder2_images, output_gif_path, duration=0.2, resize_scale=None, frame_skip=1):
    # Create a list to store image frames
    frames = []

    # Create a progress bar using tqdm
    total_images = len(folder1_images[::frame_skip])
    progress_bar = tqdm(total=total_images, desc="Creating GIF", unit="frame")

    args_list = [(img1, img2, resize_scale) for img1, img2 in zip(folder1_images[::frame_skip], folder2_images[::frame_skip])]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for combined_frame in executor.map(append_vertical_frame, args_list):
            frames.append(combined_frame)
            progress_bar.update(1)  # Increment the progress bar

    progress_bar.close()  # Close the progress bar

    # Write the frames to the GIF file with higher compression and custom frame rate
    imageio.mimsave(output_gif_path, frames, duration=duration)

    print(f"Vertical GIF created: {output_gif_path}")

if __name__ == "__main__":
    folder1 = "path/to/first_folder"
    folder2 = "path/to/second_folder"
    output_gif_path = "output_vertical.gif"

    folder1_images = [os.path.join(folder1, img_name) for img_name in os.listdir(folder1)]
    folder2_images = [os.path.join(folder2, img_name) for img_name in os.listdir(folder2)]

    if len(folder1_images) == 0 or len(folder2_images) == 0:
        print("One or both folders do not contain any images.")
    elif len(folder1_images) != len(folder2_images):
        print("The number of images in the two folders must be the same.")
    else:
        # Set options for reducing file size (comment out any option you don't want to use)
        resize_scale = 1.0  # Set to a value less than 1.0 to resize the images, or 1.0 to keep the original size
        frame_skip = 1  # Set to 1 to include all frames, set to a higher value to skip frames

        create_vertical_gif(folder1_images, folder2_images, output_gif_path, resize_scale=resize_scale, frame_skip=frame_skip)
