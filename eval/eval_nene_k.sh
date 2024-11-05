if [ $# -eq 0 ]; then
  echo "Please provide the device to use (e.g., cuda, cpu)"
  exit 1
fi

device="cuda:GPU-f09680a9-291c-2a62-8f31-ea53013587d0"

export CUDA_VISIBLE_DEVICES=GPU-f09680a9-291c-2a62-8f31-ea53013587d0
python3 eval_fix.py --model nene-v3 --index_user nene-user --index_name ayachi-nene --character Ayachi Nene --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileNene --eval_dataset nene_k.json --output_report nene-report-knowledge.json --device $device
