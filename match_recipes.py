import pandas as pd
from ultralytics import YOLO  # Assuming YOLOv11 is accessed this way
import os

# Load YOLO model
model = YOLO("yolov11n.pt")  # Replace with the correct model path if different

# Load recipe dataset
recipe_df = pd.read_csv("path/to/recipe_dataset.csv")  # Update with actual path


# Function to retrieve recipes based on the detected label
def get_recipes_for_label(label):
    # Search for recipes that include the label in their title (case insensitive)
    matches = recipe_df[recipe_df['title'].str.contains(label, case=False)]

    if matches.empty:
        return f"No recipes found for {label}."
    else:
        return matches[['title', 'ingredients']].to_dict(orient="records")


# Main function to perform training and recipe retrieval
def main():
    # Example of training (if needed)
    model.train(data="data.yaml", epochs=10, imgsz=640, project="output", name="train_results")

    # Run detection
    results = model.predict("path/to/image.jpg")  # Update with actual image path

    # Example to process detection results
    for result in results:
        detected_label = result['name']  # Adjust if result format differs
        print(f"Detected label: {detected_label}")

        # Retrieve recipes based on label
        recipes = get_recipes_for_label(detected_label)
        print(f"Recipes for {detected_label}: {recipes}")


if __name__ == "__main__":
    main()
