
import argparse
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, concatenate_datasets, load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments, BitsAndBytesConfig
from tqdm import tqdm
import torch
import numpy as np
from typing import Dict
import os
import json

BASE_CHECKPOINT_DIR = "/workspace/kann-ai/checkpoint/"
BASE_MODEL_DIR = "/workspace/kann-ai/model/"
BASE_OUTPUT_DIR = "/workspace/kann-ai/lora/"
EVAL_DATASET_DIR = "/workspace/kann-ai/eval/datasets/"
EVAL_REPORT_DIR = "/workspace/kann-ai/eval/report/"
BASEMODEL_DIR = "./basemodel/"

def load_model(model_dir, from_checkpoint):
    with torch.no_grad():
        lora_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = BASE_OUTPUT_DIR + model_dir if not from_checkpoint else BASE_CHECKPOINT_DIR + model_dir,
            max_seq_length = 8192,
            dtype = None,
            load_in_4bit = True,
            device_map = "cuda"
        )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    )
    FastLanguageModel.for_inference(lora_model)
    return lora_model, tokenizer

def main():
    for model_dir in tqdm(["nene-v4", "tsumugi-v4", "kanna-v4", "natsume-v4"]):
        lora_model, tokenizer = load_model(model_dir, False)
        lora_model.push_to_hub("alfonsusrr/" + model_dir, token="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")

if __name__ == "__main__":
    main()