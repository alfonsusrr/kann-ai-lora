#!/bin/bash

# List of scripts to run
scripts=("eval_kanna.sh", "eval_nene.sh", "eval_tsumugi.sh", "eval_natsume.sh")

# Loop through each script in sequence
for script in "${scripts[@]}"; do
  while true; do
    # Run the script with "cuda" argument and capture its output
    output=$(bash "$script" cuda)

    # Check if the output contains "ALL_DONE"
    if [[ $output == *"ALL_DONE"* ]]; then
      echo "$script finished successfully with 'ALL_DONE'"
      break  # Exit the loop and proceed to the next script
    else
      echo "$script did not finish. Retrying..."
    fi

    # Optional: add a short delay to avoid rapid retries
    sleep 2
  done
done

echo "All scripts have completed successfully."

