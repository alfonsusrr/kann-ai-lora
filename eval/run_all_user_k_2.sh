if [ $# -eq 0 ]; then
  echo "Please provide the device to use (e.g., cuda, cpu)"
  exit 1
fi

device=$1

scripts=(
    "python3 eval_fix_tmp.py --model kanna-v4 --index_user kanna-user --index_name akizuki-kanna --character Akizuki Kanna --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileKanna --eval_dataset user_knowledge_data.json --output_report kanna-report-user-knowledge-2.json --checkpoint kanna-user-knowledge.txt --user_know_eval --user_convo user_convo.json --device $device", 
)

export CUDA_VISIBLE_DEVICES=$1
for script in "${scripts[@]}"; do
  while true; do
    # Run the script with "cuda" argument and capture its output
    output=$(eval $script)

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


