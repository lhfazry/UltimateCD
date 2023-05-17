import argparse
from PIL import Image
import os

def resize_images(input_folder, output_folder, new_size):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image and resize it
            with Image.open(image_path) as image:
                resized_image = image.resize(new_size)

                # Save the resized image to the output folder
                resized_image.save(output_path)

    print("Image resizing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images in a folder.")
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("output_folder", help="Path to the output folder")
    parser.add_argument("--width", type=int, default=512, help="Width of the resized images")
    parser.add_argument("--height", type=int, default=512, help="Height of the resized images")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    new_size = (args.width, args.height)

    resize_images(input_folder, output_folder, new_size)
