import os
import argparse
from PIL import Image

def convert_images(input_folder, old_format, target_format):
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.{}'.format(old_format)):
                file_path = os.path.join(root, filename)
                try:
                    image = Image.open(file_path)
                    new_filename = os.path.splitext(filename)[0] + '.' + target_format
                    new_file_path = os.path.join(root, new_filename)
                    image.save(new_file_path, target_format)
                    os.remove(file_path)
                    print(f"Converted {filename} to {target_format} and deleted previous format.")
                except Exception as e:
                    print(f"Failed to convert {filename}: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Converter')
    parser.add_argument('input_folder', help='Path to the input folder')
    parser.add_argument('old_format', help='Current image format')
    parser.add_argument('target_format', help='Target image format')
    args = parser.parse_args()

    input_folder = args.input_folder
    old_format = args.old_format
    target_format = args.target_format

    convert_images(input_folder, old_format, target_format)
