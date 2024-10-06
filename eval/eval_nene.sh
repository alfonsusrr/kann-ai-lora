if [ $# -eq 0 ]; then
  echo "Please provide the device to use (e.g., cuda, cpu)"
  exit 1
fi

device=$1

export CUDA_VISIBLE_DEVICES=$device
python3 eval.py --model nene-v1 --index_user nene-user --index_name ayachi-nene --character Ayachi Nene --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileNene --eval_dataset nene.json --output_report nene-report.txt --device 2
