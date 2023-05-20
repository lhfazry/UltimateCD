import os
import sys
from PIL import Image

def convert_images(input_folder, target_format):
    for filename in os.listdir(input_folder):
        if filename.endswith('.{}'.format(target_format)):
            continue

        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            try:
                image = Image.open(file_path)
                new_filename = os.path.splitext(filename)[0] + '.' + target_format
                new_file_path = os.path.join(input_folder, new_filename)
                image.save(new_file_path, target_format)
                os.remove(file_path)
                print(f"Converted {filename} to {target_format} and deleted previous format.")
            except Exception as e:
                print(f"Failed to convert {filename}: {str(e)}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python image_converter.py input_folder format")
    else:
        input_folder = sys.argv[1]
        target_format = sys.argv[2]
        convert_images(input_folder, target_format)
