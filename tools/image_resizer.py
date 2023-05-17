import argparse
import os
import glob
from PIL import Image

def resize_images(input_folder, output_folder, new_size, replace):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder) and output_folder is not None:
        os.makedirs(output_folder)

    # Find all image files in the input folder and its subdirectories
    image_files = glob.glob(os.path.join(input_folder, '**', '*.*'), recursive=True)
    image_extensions = ('.jpg', '.jpeg', '.png')

    # Iterate over the image files
    for image_path in image_files:
        if image_path.lower().endswith(image_extensions):
            # Determine the output path
            if replace:
                output_path = image_path  # Replace the original image
            else:
                relative_path = os.path.relpath(image_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)

            # Open the image and resize it
            with Image.open(image_path) as image:
                resized_image = image.resize(new_size)

                # Save the resized image
                resized_image.save(output_path)

    print("Image resizing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images in a folder.")
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("--output_folder", help="Path to the output folder")
    parser.add_argument("--width", type=int, default=300, help="Width of the resized images")
    parser.add_argument("--height", type=int, default=300, help="Height of the resized images")
    parser.add_argument("--replace", action="store_true", help="Replace the original images")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    new_size = (args.width, args.height)
    replace = args.replace

    resize_images(input_folder, output_folder, new_size, replace)
