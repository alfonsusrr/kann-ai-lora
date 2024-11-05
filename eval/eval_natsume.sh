if [ $# -eq 0 ]; then
  echo "Please provide the device to use (e.g., cuda, cpu)"
  exit 1
fi

device="cuda:GPU-cb23ea95-22dc-ab66-bb1f-80e5d475a925"

export CUDA_VISIBLE_DEVICES=GPU-cb23ea95-22dc-ab66-bb1f-80e5d475a925
python3 eval_fix.py --model natsume-v4 --index_user natsume-user --index_name shiki-natsume --character Shiki Natsume --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileNatsume --eval_dataset natsume.json --output_report natsume-report.json --checkpoint natsume-cpt.txt --device $device

echo "ALL_DONE_STELLA"
