import json

def calculate_category_scores(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extract the detail_scores array
    detail_scores = data.get("detail_scores", [])

    # Initialize a dictionary to hold cumulative scores for each category
    category_scores = {}

    # Iterate over each item in detail_scores to accumulate scores
    for item in detail_scores:
        if isinstance(item, dict):  # Ensure each item is a dictionary
            scores = item.get("scores", {})  # Safely get "scores" key if it exists
            if isinstance(scores, dict):  # Ensure "scores" is a dictionary
                for category, score in scores.items():
                    category_scores[category] = category_scores.get(category, 0) + score
        else:
            print(f"Skipping invalid item: {item}")

    # Calculate percentages for each category
    category_percentages = {k: v / 25 for k, v in category_scores.items()}

    return category_scores, category_percentages

# Example usage
file_path = "./report/nene-report-gknowledge-percentage-2.json"
category_scores, category_percentages = calculate_category_scores(file_path)

# Print results
print("Category Scores:")
for category, score in category_scores.items():
    print(f"{category}: {score:.2f}")

print("\nCategory Percentages (/25):")
for category, percentage in category_percentages.items():
    print(f"{category}: {percentage:.2f}")
