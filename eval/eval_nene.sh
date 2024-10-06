read device

export CUDA_VISIBLE_DEVICES=$device
python3 eval.py --model nene-v1 --index_user nene-user --index_name ayachi-nene --character Ayachi Nene --embed_model mxbai-embed-large-v1 --modelfile_name ModelfileNene --eval_dataset nene.json --output_report nene-report.txt --device 2
