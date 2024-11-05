if [ $# -eq 0 ]; then
  echo "Please provide the device to use (e.g., cuda, cpu)"
  exit 1
fi

device="cuda:GPU-f09680a9-291c-2a62-8f31-ea53013587d0"

export CUDA_VISIBLE_DEVICES=GPU-f09680a9-291c-2a62-8f31-ea53013587d0
python3 eval_fix_stella.py --model kanna-v4 --index_user kanna-user --index_name akizuki-kanna --character Akizuki Kanna --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileKanna --eval_dataset kanna.json --output_report kanna-report.json --device $device
