if [ $# -eq 0 ]; then
  echo "Please provide the device to use (e.g., cuda, cpu)"
  exit 1
fi

device=$1

scripts=(
    "python3 eval_fix.py --model kanna-v4 --index_user kanna-user --index_name akizuki-kanna --character Akizuki Kanna --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileKanna --eval_dataset general_k.json --output_report kanna-report-gknowledge.json --checkpoint kanna-gknowledge-cpt.txt --know_eval --device $device", 
    "python3 eval_fix.py --model natsume-v4 --index_user natsume-user --index_name shiki-natsume --character Shiki Natsume --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileNatsume --eval_dataset general_k.json --output_report natsume-report-gknowledge.json --checkpoint natsume-gknowledge-cpt.txt --know_eval --device $device",
    "python3 eval_fix.py --model nene-v4 --index_user nene-user --index_name ayachi-nene --character Ayachi Nene --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileNene --eval_dataset general_k.json --output_report nene-report-gknowledge.json --checkpoint nene-gknowledge-cpt.txt --know_eval --device $device",
    "python3 eval_fix.py --model tsumugi-v4 --index_user tsumugi-user --index_name shiiba-tsumugi --character Shiiba Tsumugi --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileTsumugi --eval_dataset general_k.json --output_report tsumugi-report-gknowledge.json --checkpoint tsumugi-gknowledge-cpt.txt --know_eval --device $device"
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



