import argparse
import json
import os
from groq import Groq
import dotenv
from tqdm import tqdm
import random

dotenv.load_dotenv()

BASE_REPORT_DIR = "./report/"
CHARACTERS = ["kanna", "natsume", "nene", "tsumugi"]
VIRTUAL_KNOWLEDGE_FNAME = "-report-knowledge.json"
GENERAL_KNOWLEDGE_FNAME = "-report-gknowledge.json"

VIRTUAL_KNOWLEDGE_FNAME_SCORE = "-report-knowledge-score.json"
GENERAL_KNOWLEDGE_FNAME_SCORE = "-report-gknowledge-score.json"

def judge_correctness(correct_answer, generated):
    prompt = f'The correct answer is {correct_answer}' \
             f'Please provide a score from 0 or 1, where 0 means the answer is completely wrong, and 1 means the answer is completely correct. ' \
                f'Please provide a score for each of the following generated answers: \n\n' \
                f'1. {generated["lora_rag"]}\n' \
                f'2. {generated["lora"]}\n' \
                f'3. {generated["base_rag"]}\n' \
                f'4. {generated["base"]}\n'\
                f'Only output the score in the following format without any further comments!:'\
                f'<score>lora_rag_score, lora_score, base_rag_score, base_score</score>\n'\
                f'For example, if you think the first answer is correct, the second answer is wrong, the third answer is correct, and the fourth answer is wrong, you should output the following:\n'\
                f'<score>1, 0, 1, 0</score>'
    
    while True:
        idx = random.randint(1, 3)
        api_key = os.getenv("API_KEY_" + str(idx))
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant to judge the correctness of the following question and answer pair."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            # model="llama-3.1-70b-versatile"
            model="llama-3.1-70b-versatile"
        )

        try:
            # print(chat_completion.choices[0].message.content)
            scores_str = chat_completion.choices[0].message.content.split("<score>")[1].split("</score>")[0]
            scores_int = scores_str.split(", ")
            scores = {
                "lora_rag": int(scores_int[0]),
                "lora": int(scores_int[1]),
                "base_rag": int(scores_int[2]),
                "base": int(scores_int[3])
            }
            break
        except:
            continue
    return scores 

def judge_knowledge_character(character, args):    
    if args.general_knowledge:
        detail_scores = []
        aggregate_scores = [{
            "lora_rag": 0,
            "lora": 0,
            "base_rag": 0,
            "base": 0
        }]

        with open(BASE_REPORT_DIR + character + GENERAL_KNOWLEDGE_FNAME, "r") as f:
            question_pairs = json.load(f)
        
        for question in tqdm(question_pairs):
            correct_answer = question["expected"]
            generated = {
                "lora_rag": question["generated"],
                "lora": question["generated_no_rag"],
                "base_rag": question["generated_ollama_with_rag"],
                "base": question["generated_ollama"]
            }

            scores = judge_correctness(correct_answer, generated)
            detail_scores.append({
                "question": question["input"][0]["value"],
                "correct_answer": correct_answer,
                "generated": generated,
                "scores": scores
            })

            aggregate_scores[0]["lora_rag"] += scores["lora_rag"]
            aggregate_scores[0]["lora"] += scores["lora"]
            aggregate_scores[0]["base_rag"] += scores["base_rag"]
            aggregate_scores[0]["base"] += scores["base"]

        aggregate_scores[0]["lora_rag"] = aggregate_scores[0]["lora_rag"] / len(question_pairs)
        aggregate_scores[0]["lora"] = aggregate_scores[0]["lora"] / len(question_pairs)
        aggregate_scores[0]["base_rag"] = aggregate_scores[0]["base_rag"] / len(question_pairs)
        aggregate_scores[0]["base"] = aggregate_scores[0]["base"] / len(question_pairs)

        judge_result = {
            "detail_scores": detail_scores,
            "aggregate_scores": aggregate_scores
        }

        with open(BASE_REPORT_DIR + character + GENERAL_KNOWLEDGE_FNAME_SCORE, "w") as f:
            json.dump(judge_result, f, indent=4)

    if args.virtual_knowledge:
        detail_scores = []
        aggregate_scores = [{
            "lora_rag": 0,
            "lora": 0,
            "base_rag": 0,
            "base": 0
        }]

        with open(BASE_REPORT_DIR + character + VIRTUAL_KNOWLEDGE_FNAME, "r") as f:
            question_pairs = json.load(f)
        
        for question in tqdm(question_pairs):
            correct_answer = question["expected"]
            generated = {
                "lora_rag": question["generated"],
                "lora": question["generated_no_rag"],
                "base_rag": question["generated_ollama_with_rag"],
                "base": question["generated_ollama"]
            }

            scores = judge_correctness(correct_answer, generated)
            detail_scores.append({
                "question": question["input"][0]["value"],
                "correct_answer": correct_answer,
                "generated": generated,
                "scores": scores
            })

            aggregate_scores[0]["lora_rag"] += scores["lora_rag"]
            aggregate_scores[0]["lora"] += scores["lora"]
            aggregate_scores[0]["base_rag"] += scores["base_rag"]
            aggregate_scores[0]["base"] += scores["base"]

        aggregate_scores[0]["lora_rag"] = aggregate_scores[0]["lora_rag"] / len(question_pairs)
        aggregate_scores[0]["lora"] = aggregate_scores[0]["lora"] / len(question_pairs)
        aggregate_scores[0]["base_rag"] = aggregate_scores[0]["base_rag"] / len(question_pairs)
        aggregate_scores[0]["base"] = aggregate_scores[0]["base"] / len(question_pairs)

        judge_result = {
            "detail_scores": detail_scores,
            "aggregate_scores": aggregate_scores
        }

        with open(BASE_REPORT_DIR + character + VIRTUAL_KNOWLEDGE_FNAME_SCORE, "w") as f:
            json.dump(judge_result, f, indent=4)
    return

def judge_knowledge_result(args):
    if args.all_character:
        for character in CHARACTERS:
            print(f"Judging knowledge for character: {character}")
            judge_knowledge_character(character, args)
    else:
        for character in args.character:
            print(f"Judging knowledge for character: {character}")
            judge_knowledge_character(character, args)
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", dest='all_character', action='store_true')
    parser.add_argument("--general", dest='general_knowledge', required=False, action='store_true')
    parser.add_argument("--virtual", dest='virtual_knowledge', required=False, action='store_true')
    parser.add_argument("--character", type=str, nargs="+", required=False)

    parser.set_defaults(all_character=False)
    parser.set_defaults(general_knowledge=False)
    parser.set_defaults(virtual_knowledge=False)
    args = parser.parse_args()

    judge_knowledge_result(args)

if __name__ == "__main__":
    main()