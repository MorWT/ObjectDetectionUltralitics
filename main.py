import os
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import platform


if platform.system() == 'Windows':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Windows. Using device: {device}")

# For macOS
elif platform.system() == 'Darwin':
    if torch.backends.mps.is_available():
        device = 'mps'  # Metal Performance Shaders (for Apple Silicon GPUs)
        print("Running on macOS with Apple Silicon. Using device: MPS (Apple GPU)")
    else:
        device = 'cpu'
        print("Running on macOS. No MPS support, using CPU.")

else:
    device = 'cpu'
    print(f"Unsupported operating system. Using CPU.")


# Load configuration parameters
def load_params(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


params = load_params('config/config_params.yaml')

# Paths from params.yaml
DATA_YAML_PATH = params['data']['data_yaml_path']  # Path to data.yaml
OUTPUT_SPLIT_DATA_PATH = params['data']['output_split_data_path']
OUTPUT_PATH = params['data']['output_path']

# Check if output directory exists, if not, create it
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Step 1: Initialize the YOLO Model
# model = YOLO('yolo11n.pt').to(device)
model = YOLO('yolov8s.pt').to(device)


# Step 2: Load the Dataset
# YOLO model will automatically use paths in data.yaml for training and validation
def train_model():
    # Train the model using parameters in data.yaml
    model.train(data=DATA_YAML_PATH, epochs=10, imgsz=640, project=OUTPUT_PATH, name='train_results', nms=True,
                pretrained=True)


# Step 3: Perform Object Detection on Images
def detect_on_images(image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for image_name in tqdm(os.listdir(image_folder), desc="Processing Images"):
        image_path = os.path.join(image_folder, image_name)
        if image_path.endswith(('.jpg', '.jpeg', '.png')):
            results = model.predict(source=image_path, save=True, project=output_folder)
            print(f"Detections saved for {image_name} in {output_folder}")


# # Step 4: Perform Object Detection on Videos
# def detect_on_video(video_path, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     video_name = Path(video_path).stem
#     output_video_path = os.path.join(output_folder, f"{video_name}_output.mp4")
#     results = model.predict(source=video_path, save=True, project=output_folder)
#     print(f"Processed video saved as {output_video_path}")


# Step 5: Answer Research Questions Through Experimentation
# This could involve logging model performance or interpreting YOLO’s confidence scores on specific classes.
# Step 5: Analyze Validation Results
def analyze_results():
    # Run validation and capture results
    results = model.val(nms=True)  # Validate the model and get metrics

    # Explore the available attributes in results
    print("Available result attributes:", dir(results))

    # Access specific metrics if they exist, like precision, recall, mAP, etc.
    if hasattr(results, 'box'):
        box_metrics = results.box.mean_results()  # Retrieve mean detection metrics
        print(f"Box detection metrics (precision, recall, mAP, etc.): {box_metrics}")
    else:
        print("No box metrics available.")

    # Additional attributes to explore, such as speed and names, if relevant to your analysis
    if hasattr(results, 'speed'):
        print(f"Speed metrics: {results.speed}")
    if hasattr(results, 'maps'):
        print(f"mAP metrics: {results.maps}")


# Optional: Save Processed Video with Detections (using OpenCV for additional customization)
# def save_processed_video(input_video_path, output_video_path):
#     cap = cv2.VideoCapture(input_video_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Perform object detection on each frame
#         results = model.predict(frame)
#         annotated_frame = np.squeeze(results.render())  # Get annotated frame
#         out.write(annotated_frame)
#
#     cap.release()
#     out.release()
#     print(f"Processed video saved at {output_video_path}")


# Run steps
if __name__ == "__main__":
    # Train the model
    print("Training the model...")
    train_model()

    # Image detection
    test_images_folder = os.path.join(OUTPUT_SPLIT_DATA_PATH, 'test')
    print("Detecting objects on test images...")
    detect_on_images(test_images_folder, os.path.join(OUTPUT_PATH, 'image_detections'))

    # # Video detection (if there are test videos)
    # test_video_path = '/path/to/test_video.mp4'  # Update with actual test video path
    # print("Detecting objects on test video...")
    # detect_on_video(test_video_path, os.path.join(OUTPUT_PATH, 'video_detections'))

    # Analyze results
    print("Analyzing results...")
    analyze_results()

    # # Optional: Save processed video with detections
    # processed_video_output_path = os.path.join(OUTPUT_PATH, 'processed_video.mp4')
    # save_processed_video(test_video_path, processed_video_output_path)

    print("All steps completed.")
