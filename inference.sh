if [ $# -eq 0 ]; then
  echo "Please provide the device to use (e.g., cuda, cpu)"
  exit 1
fi

device=$1
export CUDA_VISIBLE_DEVICES=$device

python3 inference3.py --character Shiiba Tsumugi  --model tsumugi-v3 --index_name shiiba-tsumugi --index_user tsumugi-user --embed_model mxbai-embed-large-v1 --device $device
