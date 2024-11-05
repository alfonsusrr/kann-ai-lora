if [ $# -eq 0 ]; then
  echo "Please provide the device to use (e.g., cuda, cpu)"
  exit 1
fi

device="cuda:GPU-9a68e22e-6ff3-4ef9-b2ec-ef5fa6efd75c"

export CUDA_VISIBLE_DEVICES=GPU-9a68e22e-6ff3-4ef9-b2ec-ef5fa6efd75c
python3 eval_fix.py --model tsumugi-v3 --index_user tsumugi-user --index_name shiiba-tsumugi --character Shiiba Tsumugi --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileTsumugi --eval_dataset tsumugi.json --output_report tsumugi-report.txt --checkpoint tsumugi-cpt.txt --device $device
