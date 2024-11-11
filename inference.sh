if [ $# -eq 0 ]; then
  echo "Please provide the device to use (e.g., cuda, cpu)"
  exit 1
fi

device="cuda:GPU-a4210910-5df3-9fdb-00d5-205dc94dc6e9"
export CUDA_VISIBLE_DEVICES=GPU-a4210910-5df3-9fdb-00d5-205dc94dc6e9

python3 inference3.py --character Shiki Natsume  --model natsume-v4 --index_name shiki-natsume --index_user natsume-user --embed_model mxbai-embed-large-v1 --device $device
