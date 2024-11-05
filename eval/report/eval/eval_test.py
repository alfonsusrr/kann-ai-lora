import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Sample function to extract the string after each label
def extract_data(text, label):
    pattern = rf"{label}:(.*?)(?=(Generated|\(.*?\)|$))"  # Matches until 'Generated', label in parenthesis or end
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

# Read the input file from the specified path
with open("../nene-report.txt", "r") as file:
    data = file.read()

# Split by message sections
messages = data.split("--------------------------------------------------")

# Initialize scorer for ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Initialize accumulators for BLEU and ROUGE scores
bleu_scores = {'Generated': [], 'no_rag': [], 'ollama': [], 'ollama_rag': []}
cumulative_bleu_scores = {'Generated': [], 'no_rag': [], 'ollama': [], 'ollama_rag': []}
rouge_scores = {'Generated': {'rouge1': [], 'rouge2': [], 'rougeL': []},
                'no_rag': {'rouge1': [], 'rouge2': [], 'rougeL': []},
                'ollama': {'rouge1': [], 'rouge2': [], 'rougeL': []},
                'ollama_rag': {'rouge1': [], 'rouge2': [], 'rougeL': []}}

# Initialize counters for zero scores
zero_bleu_counts = {variant: 0 for variant in bleu_scores.keys()}
zero_rouge_counts = {variant: 0 for variant in rouge_scores.keys()}

# Create a smoothing function for BLEU
smoothing_function = SmoothingFunction().method1  # Apply smoothing to avoid zero precision for higher n-grams

# String to accumulate the final report output
final_report = ""

# Open the iteration report file for writing
with open("iteration_report.txt", "w") as iteration_file:
    # Iterate through each message block and extract data
    for idx, message in enumerate(messages):
        if message.strip():  # Skip empty sections
            iteration_file.write(f"Processing Message {idx + 1}\n")
            iteration_file.write(f"Processed string: {message}\n")  # Add processed string

            # Extract different sections
            input_text = extract_data(message, "Input")
            expected_text = extract_data(message, "Expected")
            generated_text = extract_data(message, "Generated")
            generated_no_rag = extract_data(message, "Generated (no RAG)")
            generated_ollama = extract_data(message, "Generated (Ollama)")
            generated_ollama_rag = extract_data(message, "Generated (Ollama with RAG)")

            # List of all versions for processing
            variants = {
                "Generated": generated_text,
                "no_rag": generated_no_rag,
                "ollama": generated_ollama,
                "ollama_rag": generated_ollama_rag
            }

            # Calculate scores for each variant
            for variant_name, variant_text in variants.items():
                # BLEU for the variant
                bleu_score = sentence_bleu([expected_text.split()], variant_text.split())
                bleu_scores[variant_name].append(bleu_score)

                # Cumulative BLEU for the variant
                cumulative_bleu_score = sentence_bleu(
                    [expected_text.split()],
                    variant_text.split(),
                    smoothing_function=smoothing_function
                )
                cumulative_bleu_scores[variant_name].append(cumulative_bleu_score)

                # ROUGE for the variant
                rouge_score = scorer.score(expected_text, variant_text)
                rouge_scores[variant_name]['rouge1'].append(rouge_score['rouge1'].fmeasure)
                rouge_scores[variant_name]['rouge2'].append(rouge_score['rouge2'].fmeasure)
                rouge_scores[variant_name]['rougeL'].append(rouge_score['rougeL'].fmeasure)

                # Increment zero score counters
                if bleu_score == 0:
                    zero_bleu_counts[variant_name] += 1
                if rouge_score['rouge1'].fmeasure == 0 and rouge_score['rouge2'].fmeasure == 0 and rouge_score['rougeL'].fmeasure == 0:
                    zero_rouge_counts[variant_name] += 1

                # Write iteration scores to the file
                iteration_file.write(f"{variant_name} BLEU score: {bleu_score}\n")
                iteration_file.write(f"{variant_name} Cumulative BLEU score: {cumulative_bleu_score}\n")
                iteration_file.write(f"{variant_name} ROUGE scores: {rouge_score}\n\n")

# Calculate final average scores for all variants
final_report += "Final Report:\n"
total_messages = len(messages)
for variant_name in variants.keys():
    # Final BLEU score
    final_bleu_score = sum(bleu_scores[variant_name]) / len(bleu_scores[variant_name])

    # Final Cumulative BLEU score
    final_cumulative_bleu_score = sum(cumulative_bleu_scores[variant_name]) / len(cumulative_bleu_scores[variant_name])

    # Final ROUGE scores
    final_rouge_score = {
        "rouge1": sum(rouge_scores[variant_name]["rouge1"]) / len(rouge_scores[variant_name]["rouge1"]),
        "rouge2": sum(rouge_scores[variant_name]["rouge2"]) / len(rouge_scores[variant_name]["rouge2"]),
        "rougeL": sum(rouge_scores[variant_name]["rougeL"]) / len(rouge_scores[variant_name]["rougeL"])
    }

    # Calculate percentages of zero scores
    zero_bleu_percentage = (zero_bleu_counts[variant_name] / total_messages) * 100
    zero_rouge_percentage = (zero_rouge_counts[variant_name] / total_messages) * 100

    # Append final scores to the final report
    final_report += f"\n{variant_name} Final BLEU score: {final_bleu_score}\n"
    final_report += f"{variant_name} Final Cumulative BLEU score: {final_cumulative_bleu_score}\n"
    final_report += f"{variant_name} Final ROUGE scores: {final_rouge_score}\n"
    final_report += f"{variant_name} Percentage of zero BLEU scores: {zero_bleu_percentage:.2f}%\n"
    final_report += f"{variant_name} Percentage of zero ROUGE scores: {zero_rouge_percentage:.2f}%\n"

# Write the final report to a file
with open("final_report.txt", "w") as report_file:
    report_file.write(final_report)

print("Iteration results written to 'iteration_report.txt' and final report written to 'final_report.txt'")

