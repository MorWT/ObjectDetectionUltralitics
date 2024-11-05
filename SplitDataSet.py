import os
import random
import yaml
from PIL import Image
from utils import load_params

# Load parameters from params.yaml
params = load_params('config/config_params.yaml')

# Paths from params.yaml
DATASET_PATH = params['data']['dataset_path']
OUTPUT_PATH = params['data']['output_split_data_path']
LABELS_PATH = params['data']['labels_path']
DATA_YAML_PATH = params['data']['data_yaml_path']

# Ensure output directories exist
os.makedirs(os.path.join(OUTPUT_PATH, 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'val'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'test'), exist_ok=True)


# Function to split dataset
def split_dataset(dataset_path):
    food_types = os.listdir(dataset_path)  # List of food type folders
    for food_type in food_types:
        food_type_path = os.path.join(dataset_path, food_type)
        if os.path.isdir(food_type_path):
            images = os.listdir(food_type_path)
            random.shuffle(images)  # Shuffle images for randomness

            # Calculate split indices using loaded ratios
            train_size = int(len(images) * params['split_ratios']['train'])
            val_size = int(len(images) * params['split_ratios']['val'])

            # Split images
            train_images = images[:train_size]
            val_images = images[train_size:train_size + val_size]
            test_images = images[train_size + val_size:]

            # Helper function to save images as JPEG
            def save_as_jpeg(src_path, dest_path):
                with Image.open(src_path) as img:
                    img = img.convert("RGB")  # Ensure RGB format
                    img.save(dest_path, format="JPEG")

            # Move images to respective directories
            for img_name in train_images:
                src = os.path.join(food_type_path, img_name)
                dest = os.path.join(OUTPUT_PATH, 'train', food_type)
                os.makedirs(dest, exist_ok=True)
                save_as_jpeg(src, os.path.join(dest, img_name))

            for img_name in val_images:
                src = os.path.join(food_type_path, img_name)
                dest = os.path.join(OUTPUT_PATH, 'val', food_type)
                os.makedirs(dest, exist_ok=True)
                save_as_jpeg(src, os.path.join(dest, img_name))

            for img_name in test_images:
                src = os.path.join(food_type_path, img_name)
                dest = os.path.join(OUTPUT_PATH, 'test', food_type)
                os.makedirs(dest, exist_ok=True)
                save_as_jpeg(src, os.path.join(dest, img_name))



# Execute the split
split_dataset(DATASET_PATH)
print("Dataset split into train, val, and test folders.")
print("data.yaml file created successfully.")
