import os
import shutil
import random

# Paths
source_dir = '..\images'  # Source directory where your images are stored
destination_dir = '..\data'  # Destination directory where train/test folders will be created
train_ratio = 0.8  # 80% of images for training, 20% for testing

# Create directories for train and test
train_dir = os.path.join(destination_dir, 'train')
test_dir = os.path.join(destination_dir, 'test')

# Create train and test folders if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all fruit categories (folder names under source_dir)
fruit_names = os.listdir(source_dir)

for fruit_name in fruit_names:
    # Path to the fruit folder
    fruit_folder = os.path.join(source_dir, fruit_name)

    if not os.path.isdir(fruit_folder):
        continue  # Skip if it's not a directory

    # Get all image files in the fruit folder
    image_files = os.listdir(fruit_folder)

    # Shuffle image files to randomize
    random.shuffle(image_files)

    # Calculate the split index for training and testing
    split_index = int(len(image_files) * train_ratio)
    
    # Training images and directories
    train_images = image_files[:split_index]
    train_fruit_dir = os.path.join(train_dir, fruit_name)
    os.makedirs(train_fruit_dir, exist_ok=True)  # Create a sub-folder for each fruit

    # Copy training images
    for image_file in train_images:
        src_image_path = os.path.join(fruit_folder, image_file)
        dest_image_path = os.path.join(train_fruit_dir, image_file)
        shutil.copy(src_image_path, dest_image_path)

    # Testing images and directories
    test_images = image_files[split_index:]
    test_fruit_dir = os.path.join(test_dir, fruit_name)
    os.makedirs(test_fruit_dir, exist_ok=True)  # Create a sub-folder for each fruit

    # Copy testing images
    for image_file in test_images:
        src_image_path = os.path.join(fruit_folder, image_file)
        dest_image_path = os.path.join(test_fruit_dir, image_file)
        shutil.copy(src_image_path, dest_image_path)

print("Images have been successfully copied and split into training and testing sets!")
