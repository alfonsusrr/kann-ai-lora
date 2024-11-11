if [ $# -eq 0 ]; then
  echo "Please provide the device to use (e.g., cuda, cpu)"
  exit 1
fi

device="cuda:GPU-f09680a9-291c-2a62-8f31-ea53013587d0"

export CUDA_VISIBLE_DEVICES=GPU-f09680a9-291c-2a62-8f31-ea53013587d0
python3 eval_fix.py --model natsume-v4 --index_user natsume-user --index_name shiki-natsume --character Shiki Natsume --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileNatsume --eval_dataset natsume_k.json --output_report natsume-report-knowledge.json --checkpoint natsume-knowledge-cpt.txt --know_eval --device $device 
