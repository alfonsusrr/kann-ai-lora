if [ $# -eq 0 ]; then
  echo "Please provide the device to use (e.g., cuda, cpu)"
  exit 1
fi

device="cuda:GPU-f09680a9-291c-2a62-8f31-ea53013587d0"

export CUDA_VISIBLE_DEVICES=GPU-f09680a9-291c-2a62-8f31-ea53013587d0
python3 eval_fix.py --model tsumugi-v4 --index_user tsumugi-user --index_name shiiba-tsumugi --character Shiiba Tsumugi --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileTsumugi --eval_dataset tsumugi_k.json --output_report tsumugi-report-knowledge.json --checkpoint tsumugi-knowledge-cpt.txt --device $device 
