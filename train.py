import argparse
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, concatenate_datasets, load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

import os
import json

BASE_CHECKPOINT_DIR = "./checkpoint/"
BASE_MODEL_DIR = "./model/"
BASE_OUTPUT_DIR = "./lora/"



def set_env(device):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

def load_model(model_name, from_checkpoint, checkpoint_dir, lora_rank, lora_alpha):
    model = None
    tokenizer = None
    if from_checkpoint:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = BASE_CHECKPOINT_DIR + checkpoint_dir,
            max_seq_length = 8192,
            dtype = None,
            load_in_4bit = True
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = BASE_MODEL_DIR + model_name,
            max_seq_length = 8192,
            dtype = None,
            load_in_4bit = True
        )
    lora_model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = True,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    )
    return lora_model, tokenizer

def format_messages(raw_messages, character):
    messages = raw_messages["conversation"]
    messages = json.loads(messages)
    messages = [{
        "from": "gpt" if message["role"] in character else "human",
        "value": message["content"]
    } for message in messages]
    messages = [{
        "from": "system",
        "value": f"You are a character with aliases : {', '.join(character)}. Complete the conversation with your personality. "
    }] + messages
    return {
        "conversations": messages
    }

def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }


def process_dataset(dataset, hf_token, character, tokenizer):
    dataset = load_dataset(dataset, token=hf_token, split="train")
    dataset = dataset.map(
        format_messages,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
        fn_kwargs = {"character": character}
    )

    dataset = dataset.map(
        formatting_prompts_func,
        num_proc=os.cpu_count(),
        fn_kwargs = {"tokenizer": tokenizer}
    )
    return dataset

def train(args):
    set_env(args.device)
    lora_model, tokenizer = load_model(args.model, args.from_checkpoint, args.output_dir, args.lora_rank, args.lora_alpha)
    dataset = process_dataset(args.dataset, args.hf_token, args.character, tokenizer)

    tokenizer.padding_side = "right"
    packing = False
    trainer = SFTTrainer(
        model = lora_model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 8192,
        dataset_num_proc = os.cpu_count(),
        packing = packing, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = args.batch_size,
            auto_find_batch_size = True,
            gradient_accumulation_steps = 1,
            warmup_steps = 5,
            num_train_epochs=args.epochs,
            learning_rate = args.learning_rate, # lower lr with no packing
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "paged_adamw_8bit",
            weight_decay = 1e-12,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "BASE_CHECKPOINT_DIR" + args.output_dir,
            save_steps = 100,
            resume_from_checkpoint=args.from_checkpoint,
        ),
    )

    trainer_stats = trainer.train(resume_from_checkpoint=args.from_checkpoint)
    lora_model.save_pretrained(BASE_OUTPUT_DIR + args.output_dir) # Local saving

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--from_checkpoint", type=int, default=False)
    parser.add_argument("--character", type=str, nargs="+", required=True)
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()