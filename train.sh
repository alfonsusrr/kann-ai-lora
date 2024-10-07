if [ $# -eq 0 ]; then
  echo "Please provide the device to use (e.g., cuda, cpu)"
  exit 1
fi

device=$1
export CUDA_VISIBLE_DEVICES=$device

python3 train.py --epochs 3 --lora_rank 8 --lora_alpha 16 --character Ayachi Nene --device 2 --output_dir nene-v2 --hf_token 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 --model Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset sabbat_nene --device $device