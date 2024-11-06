import json
import random
import sys

def process_file(filename):
    try:
        # Load data from the JSON file with UTF-8 encoding
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        # Filter out items with "expected": "......"
        filtered_data = [item for item in data if item.get("expected") != "......"]
        
        # Check if there are enough items to sample
        if len(filtered_data) < 100:
            print(f"Not enough items in {filename} after filtering.")
            return
        
        # Select 25 random items from the filtered list
        random_items = random.sample(filtered_data, 100)
        
        # Save the selected items to a new JSON file
        output_filename = f"out_{filename}"
        with open(output_filename, "w", encoding="utf-8") as outfile:
            json.dump(random_items, outfile, indent=4)

        print(f"Processed {filename} -> {output_filename}")

    except UnicodeDecodeError as e:
        print(f"Error reading {filename}: {e}")

def main():
    # Ensure at least one file is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py file1.json file2.json ...")
        sys.exit(1)
    
    # Process each file provided as an argument
    for filename in sys.argv[1:]:
        process_file(filename)

if __name__ == "__main__":
    main()
