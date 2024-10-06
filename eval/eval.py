import argparse
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, concatenate_datasets, load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments, BitsAndBytesConfig
from tqdm import tqdm
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import uuid
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Dict
import nltk
from rouge_score import rouge_scorer
import os
import json
import gc
import ollama

BASE_CHECKPOINT_DIR = "/workspace/kann-ai/checkpoint/"
BASE_MODEL_DIR = "/workspace/kann-ai/model/"
BASE_OUTPUT_DIR = "/workspace/kann-ai/lora/"
EVAL_DATASET_DIR = "/workspace/kann-ai/eval/datasets/"
EVAL_REPORT_DIR = "/workspace/kann-ai/eval/report/"

load_dotenv()
nltk.download('all')

global lora_model, tokenizer, embed_model, embed_tokenizer, index, baseline_model_name
def set_env(device):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

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

def load_embed_model(embed_model):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR + embed_model, trust_remote_code=True)
        model = AutoModel.from_pretrained(BASE_MODEL_DIR + embed_model, trust_remote_code=True, quantization_config=nf4_config, device_map="cuda:0")
    return model, tokenizer


def load_ollama_model(args):
    global baseline_model_name
    
    modelfile_name = "./basemodel/" + args.modelfile_name
    
    with open(modelfile_name, 'w') as f:
        ollama.create(model=f"{modelfile_name}", modelfile=f.read())
        
    baseline_model_name = f"{modelfile_name}"

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

def formatting_prompts_func(examples):
    global tokenizer
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts }

def initialize_RAG(index_name):
    api_key = os.getenv("PINECONE_API_KEY")
    if api_key is None:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index

# Embedding Tools
def pooling(outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
    else:
        raise NotImplementedError
    last_layer = outputs.detach().cpu().numpy()
    return last_layer

def embedding_function(model, tokenizer, text):
    inputs = tokenizer(text, padding=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state
        embeddings = pooling(outputs, inputs, 'cls')

    del inputs
    del outputs

    gc.collect()
    torch.cuda.empty_cache()
    return embeddings
# --- End of Embedding Tools ---

def transform_query(query: str) -> str:
    """ For retrieval, add the prompt for query (not for documents).
    """
    return f'Represent this sentence for searching relevant passages: {query}'

def document_retrieval(model, tokenizer, pc, index_name, query):
    query_embedding = embedding_function(model, tokenizer, query)
    query_results = pc.query(
        namespace=index_name,
        vector=query_embedding[0].tolist(),
        top_k=1,
        include_metadata=True
    )
    results = []
    for result in query_results["matches"]:
        results.append(result['metadata']['text'])
    return results

def inference(args):
    set_env(args.device)
    lora_model, tokenizer = load_model(args.model, args.from_checkpoint)
    embed_model, embed_tokenizer = load_embed_model(args.embed_model)
    index = initialize_RAG(args.index_name)

    prev_messages = []
    while True:
        message_content = input("You: ")
        rag_results_list = document_retrieval(embed_model, embed_tokenizer, index, args.index_name, message_content)
        rag_results = rag_results_list[0]
        user_results = rag_results_list[1]
        rag_prompt = f"Here are some examples of how you might respond as {' or '.join(args.character) if len(args.character) > 1 else args.character[0]} based on the given context and characters: {', '.join(rag_results)} \n"

        if len(user_results) > 0:
            rag_prompt += f"Here are also some related chat history with this person: {', '.join(user_results)} \n"

        prev_messages.append({"from": "user", "value": message_content})

        appended_messages = [{
            "from": "system",
            "value": f"You are roleplaying a character that is named {' or '.join(args.character) if len(args.character) > 1 else args.character[0]}. Please provide a response that is engaging, in-character, and adds depth to the conversation. Make sure to be as detailed as possible. Do not include the character's name or any tags before the response. Only provide the spoken dialogue of the character you are roleplaying. \n" + rag_prompt
        }] + prev_messages

        text = tokenizer.apply_chat_template(
            appended_messages,
            tokenize = False,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        )

        inputs = tokenizer(text, return_tensors="pt", padding=True).to('cuda')
        output = lora_model.generate(**inputs, max_new_tokens=500, temperature=1)
        text = tokenizer.batch_decode(output)

        parsed_text_1 = text[0].split("<|end_header_id|>")[-1]
        parsed_text_2 = parsed_text_1.split("<|eot_id|>")[0].strip()
        prev_messages.append({"from": "assistant", "value": parsed_text_2})
        print(f"{args.character[0]}: {parsed_text_2}")


# LoRA + RAG
def handle_single_message(message_content, args):  
    global lora_model, tokenizer, embed_model, embed_tokenizer, index  

    rag_results_list = document_retrieval(embed_model, embed_tokenizer, index, args.index_name, message_content)
    rag_results = rag_results_list[0]
    rag_prompt = f"Here are some examples of how you might respond as {' or '.join(args.character) if len(args.character) > 1 else args.character[0]} based on the given context and characters: {', '.join(rag_results)} \n"

    appended_messages = [{
        "from": "system",
        "value": f"You are roleplaying a character that is named {' or '.join(args.character) if len(args.character) > 1 else args.character[0]}. Please provide a response that is engaging, in-character, and adds depth to the conversation. Make sure to be as detailed as possible. Do not include the character's name or any tags before the response. Only provide the spoken dialogue of the character you are roleplaying. \n" + rag_prompt
    }]

    text = tokenizer.apply_chat_template(
        appended_messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    )

    inputs = tokenizer(text, return_tensors="pt", padding=True).to('cuda')
    with torch.no_grad():
        output = lora_model.generate(**inputs, max_new_tokens=500, temperature=1)
        text = tokenizer.batch_decode(output)

    inputs.to("cpu")
    output.to("cpu")
    del inputs
    del output

    parsed_text_1 = text[0].split("<|end_header_id|>")[-1]
    parsed_text_2 = parsed_text_1.split("<|eot_id|>")[0].strip()
    
    gc.collect()
    torch.cuda.empty_cache()
    return parsed_text_2

# LoRA only
def handle_single_message_no_rag(message_content, args):
    global lora_model, tokenizer

    appended_messages = [{
        "from": "system",
        "value": f"You are roleplaying a character that is named {' or '.join(args.character) if len(args.character) > 1 else args.character[0]}. Please provide a response that is engaging, in-character, and adds depth to the conversation. Make sure to be as detailed as possible. Do not include the character's name or any tags before the response. Only provide the spoken dialogue of the character you are roleplaying. \n"
    }]

    text = tokenizer.apply_chat_template(
        appended_messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    )

    inputs = tokenizer(text, return_tensors="pt", padding=True).to('cuda')
    with torch.no_grad():
        output = lora_model.generate(**inputs, max_new_tokens=500, temperature=1)
        text = tokenizer.batch_decode(output)

    inputs.to("cpu")
    output.to("cpu")
    del inputs
    del output

    parsed_text_1 = text[0].split("<|end_header_id|>")[-1]
    parsed_text_2 = parsed_text_1.split("<|eot_id|>")[0].strip()
    
    gc.collect()
    torch.cuda.empty_cache()
    return parsed_text_2

# Ollama model only (System prompt only)
def ollama_only(message_content):
    global baseline_model_name

    response = ollama.chat(model=baseline_model_name, messages=[
        {
            'role': 'user',
            'content': message_content
        }
    ])

    return response['message']['content']

def ollama_with_rag(message_content, args):
    global baseline_model_name, embed_model, embed_tokenizer, index

    rag_results_list = document_retrieval(embed_model, embed_tokenizer, index, args.index_name, message_content)
    rag_results = rag_results_list[0]
    rag_prompt = f"Here are some examples of how you might respond as {' or '.join(args.character) if len(args.character) > 1 else args.character[0]} based on the given context and characters: {', '.join(rag_results)} \n"

    response = ollama.chat(model=baseline_model_name, messages=[
        {
            'role': 'user',
            'content': f"{message_content}\n{rag_prompt}"
        }
    ])

    return response['message']['content']

def calculate_bleu(reference, generated):
    """Calculate BLEU score between reference and generated text."""
    reference_tokens = [nltk.word_tokenize(reference)]
    generated_tokens = nltk.word_tokenize(generated)
    bleu_score = nltk.translate.bleu_score.sentence_bleu(reference_tokens, generated_tokens)
    return bleu_score

def calculate_rouge(reference, generated):
    """Calculate ROUGE score between reference and generated text."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

def evaluate_conversations(data, args):
    reference_responses = []
    generated_responses = []
    generated_responses_no_rag = []
    generated_responses_ollama = []
    generated_responses_ollama_with_rag = []
    
    global lora_model, tokenizer, embed_model, embed_tokenizer, index

    set_env(args.device)
    lora_model, tokenizer = load_model(args.model, args.from_checkpoint)
    embed_model, embed_tokenizer = load_embed_model(args.embed_model)
    index = initialize_RAG(args.index_name)

    with open(EVAL_REPORT_DIR + args.output_report, 'w', encoding='utf-8') as file:
        for i in tqdm(range(len(data))):
            if i % 10 != 0:
                continue
            # Get the reference and generated text
            conversation = data[i]
            input_message = ""
            
            file.write(f"Message length: {len(conversation['input'])}\n")
            for message in conversation['input']:
                input_message += f"{message['role']}: {message['content']}\n"
            
            reference_response = conversation['result']['content']
            generated_response_val = handle_single_message(input_message, args)
            generated_response_no_rag_val = handle_single_message_no_rag(input_message, args)
            generated_response_ollama_val = ollama_only(input_message)
            generated_response_ollama_with_rag_val = ollama_with_rag(input_message, args)
            
            # Accumulate reference and generated responses for later evaluation
            reference_responses.append(reference_response)
            generated_responses.append(generated_response_val)
            generated_responses_no_rag.append(generated_response_no_rag_val)
            generated_responses_ollama.append(generated_response_ollama_val)
            generated_responses_ollama_with_rag.append(generated_response_ollama_with_rag_val)

            # Log the individual conversation to the file
            file.write(f"Input: {input_message}\n")
            file.write(f"Expected: {reference_response}\n")
            file.write(f"Generated: {generated_response_val}\n")
            file.write(f"Generated (No RAG): {generated_response_no_rag_val}\n")
            file.write(f"Generated (Ollama): {generated_response_ollama_val}\n")
            file.write(f"Generated (Ollama with RAG): {generated_response_ollama_with_rag_val}\n")
            file.write('-' * 50 + '\n')

        # Calculate BLEU and ROUGE scores over all responses at once
        bleu_score_main = calculate_bleu(reference_responses, generated_responses)
        rouge_score_main = calculate_rouge(reference_responses, generated_responses)
        bleu_score_no_rag = calculate_bleu(reference_responses, generated_responses_no_rag)
        rouge_score_no_rag = calculate_rouge(reference_responses, generated_responses_no_rag)
        bleu_score_ollama = calculate_bleu(reference_responses, generated_responses_ollama)
        rouge_score_ollama = calculate_rouge(reference_responses, generated_responses_ollama)
        bleu_score_ollama_with_rag = calculate_bleu(reference_responses, generated_responses_ollama_with_rag)
        rouge_score_ollama_with_rag = calculate_rouge(reference_responses, generated_responses_ollama_with_rag)

        # Log the final BLEU and ROUGE scores to the file
        file.write('-' * 50 + '\n')
        file.write('Final Scores BLEU\n')
        file.write(f"Final BLEU Score Main: {bleu_score_main}\n")
        file.write(f"Final BLEU Score (No RAG): {bleu_score_no_rag}\n")
        file.write(f"Final BLEU Score (Ollama): {bleu_score_ollama}\n")
        file.write(f"Final BLEU Score (Ollama with RAG): {bleu_score_ollama_with_rag}\n")
        file.write('-' * 50 + '\n')
        file.write('Final Scores ROUGE\n')
        file.write(f"Final ROUGE Score Main: {rouge_score_main}\n")
        file.write(f"Final ROUGE Score (No RAG): {rouge_score_no_rag}\n")
        file.write(f"Final ROUGE Score (Ollama): {rouge_score_ollama}\n")
        file.write(f"Final ROUGE Score (Ollama with RAG): {rouge_score_ollama_with_rag}\n")
        file.write('-' * 50 + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=5)
    parser.add_argument("--index_name", type=str, required=True)
    parser.add_argument("--index_user", type=str, required=True)
    parser.add_argument("--modelfile_name", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--output_report", type=str, required=True)
    parser.add_argument("--embed_model", type=str, required=True)
    parser.add_argument("--character", type=str, nargs="+", required=True)
    parser.add_argument("--from_checkpoint", type=bool, default=False)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    # inference(args)
    
    with open(EVAL_DATASET_DIR + args.eval_dataset , "r") as f:
        data = json.load(f)
    
    evaluate_conversations(data, args)

if __name__ == "__main__":
    main()
