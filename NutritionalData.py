import yaml
import pandas as pd
import re
from utils import load_params


class FoodItem:
    def __init__(self, name, caloric_value, fat_value, protein_value):
        self.name = name
        self.caloric_value = caloric_value
        self.fat_value = fat_value
        self.protein_value = protein_value

    def __repr__(self):
        return (f"FoodItem(name={self.name}, "
                f"caloric_value={self.caloric_value}, "
                f"fat_value={self.fat_value}, "
                f"protein_value={self.protein_value})")


# Load the labels from the data.yaml configuration file
def load_labels(data_config):
    labels = data_config['names'].values()
    return [label.lower().replace('_', ' ') for label in labels]


# Search for labels in the name column of the CSV and average multiple entries
def match_labels_and_average(labels, nutrition_df):
    food_items = {}

    for label in labels:
        # Create regex pattern for each label to allow flexible matching
        label_pattern = label.replace(' ', r'\s*')  # Create the regex string for spaces
        pattern = re.compile(rf"\b{label_pattern}\b", re.IGNORECASE)

        # Filter rows where the name column contains the label pattern
        matches = nutrition_df[nutrition_df['name'].str.contains(pattern, na=False)]

        if not matches.empty:
            # Average the nutritional values if there are multiple rows
            avg_values = matches[['calories', 'total_fat', 'protein']].mean()  # Average only the relevant columns
            food_items[label] = FoodItem(
                name=label,
                caloric_value=avg_values['calories'],
                fat_value=avg_values['total_fat'],
                protein_value=avg_values['protein']
            )
        else:
            print(f"No match found for label: {label}")

    return food_items


# Save the results to a new file (optional)
def save_nutritional_dict(output_path, food_items):
    food_items_dict = {label: vars(item) for label, item in food_items.items()}  # Convert FoodItem instances to dict
    with open(output_path, 'w') as file:
        yaml.dump(food_items_dict, file)


# Main function to tie everything together
def main():
    # Paths
    params = load_params('config/config_params.yaml')
    csv_path = params['data']['nutritional_data']
    yaml_path = params['data']['data_yaml_path']
    output_path = params['data']['nutritional_output_path']

    # Load data
    data_config = load_params(yaml_path)
    labels = load_labels(data_config)
    nutrition_df = pd.read_csv(csv_path)

    # Filter only the relevant columns
    columns_to_keep = ['name', 'calories', 'total_fat', 'protein']
    nutrition_df = nutrition_df[columns_to_keep]

    # Match labels and compute averages
    food_items = match_labels_and_average(labels, nutrition_df)

    # Save results (optional)
    save_nutritional_dict(output_path, food_items)

    print("Nutritional data dictionary created successfully.")
    print(food_items)  # Display the dictionary for verification


# Run the script
if __name__ == "__main__":
    main()
