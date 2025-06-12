import argparse
import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
import json
import random
from tqdm import tqdm
from eval.utils import get_next_word_predictions, load_hf_tokenizer, load_hf_lm, query_openai_chat_model, dynamic_import_function, upload_results_to_hf, check_and_upload_model_metadata

choices = ["A", "B", "C", "D"]

def load_jsonl(file_path):
    """Loads a .jsonl file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def generate_prompt_from_examples(q, with_explanation=False, is_test_prompt=False):
    """
    Generates a formatted prompt from a question dictionary.
    - Shuffles the answer choices randomly.
    - `with_explanation`: If True, includes the step-by-step explanation. This is independent
      of whether it's a test prompt or a few-shot example.
    - `is_test_prompt`: If True, ends with "Answer:"; otherwise, provides the correct answer label.
    """
    correct_answer_text = q["Correct Answer"].strip()
    answers = [
        correct_answer_text,
        q["Incorrect Answer 1"].strip(),
        q["Incorrect Answer 2"].strip(),
        q["Incorrect Answer 3"].strip()
    ]
    random.shuffle(answers)
    
    shuffled_choices_dict = dict(zip(choices, answers))
    correct_label = ""
    
    # Build the question and choices part of the prompt
    output = f'What is the correct answer to this question: {q["Question"]}'
    output+='\n\nChoices:\n'
    for label, value in shuffled_choices_dict.items():
        output += f'({label}) {value}\n'
        if value == correct_answer_text:
            correct_label = label
            
    # Add explanation if requested, regardless of prompt type
    if with_explanation:
        output += f"\nLet's think step by step:\n{q['Explanation']}\n"
        
    # Add the final line based on whether it's a test prompt
    if not is_test_prompt:
        # For few-shot examples, provide the final answer.
        output += f'The correct answer is ({correct_label})\n\n'
    else:
        # For the final test question, prompt the model for the answer.
        output += f"\n\nFormat your response as follows: \"The correct answer is (insert answer here)\""
        
    return output, correct_label

@torch.no_grad()
def eval_hf_model(args, model, tokenizer, test_examples, batch_size=1):
    prompts = []
    groud_truths = []
    
    for i in tqdm(range(len(test_examples)), desc=f"Evaluating gpqa"):
        test_example = test_examples[i]
        
        train_prompt = ""

        test_prompt_part, correct_label = generate_prompt_from_examples(
            test_example, 
            with_explanation=args.with_explanation, 
            is_test_prompt=True
        )
        groud_truths.append(correct_label)
        
        prompt = train_prompt + test_prompt_part
        prompts.append(prompt)
        
    answer_choice_ids = [tokenizer.encode(" " + choice, add_special_tokens=False)[-1] for choice in choices]
    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, return_token_predictions=False, batch_size=batch_size
    )

    cors = []
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)
        
    acc = np.mean(cors)
    return cors, acc, all_probs, groud_truths

def eval_openai_chat_engine(args, engine, test_examples, batch_size=1):
    raise NotImplementedError("OpenAI evaluation needs to be refactored for the new prompt format.")

def main(args):
    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        tokenizer = load_hf_tokenizer(model_name_or_path=args.model_name_or_path, revision=args.hf_revision, tokenizer_name_or_path=args.tokenizer_name_or_path, use_fast_tokenizer=not args.use_slow_tokenizer)
        model = load_hf_lm(model_name_or_path=args.model_name_or_path, revision=args.hf_revision, load_in_8bit=args.load_in_8bit, device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto", gptq_model=args.gptq)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_examples = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=os.environ["HF_TOKEN"])
        
    if args.n_instances and args.n_instances < len(test_examples):
        test_examples = test_examples.select(range(args.n_instances))

    if args.model_name_or_path:
        cors, acc, probs, ground_truths = eval_hf_model(args, model, tokenizer, test_examples, args.eval_batch_size)
    else:
        cors, acc, probs, ground_truths = eval_openai_chat_engine(args, args.openai_engine, test_examples, args.eval_batch_size)
        
    print(f"\n--- Final Result for ---")
    print(f"Average accuracy: {acc:.3f}")

    results_data = []
    for i, test_example in enumerate(test_examples):
        result = {
            "question": test_example["Question"],
            "ground_truth_label": ground_truths[i],
            "prediction_is_correct": cors[i],
        }
        for j, choice in enumerate(choices):
            result[f"prob_{choice}"] = probs[i][j]
        results_data.append(result)

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(args.save_dir, f"results.csv"), index=False)
    print(f"Detailed results saved to {os.path.join(args.save_dir, f'results.csv')}")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump({"average_accuracy": acc}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5, help="Number of few-shot examples to use.")
    parser.add_argument("--with_explanation", action="store_true", help="Include step-by-step explanations in prompts.")
    parser.add_argument("--save_dir", type=str, default="results/")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--hf_revision", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--openai_engine", type=str, default=None)
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If given, we will use the slow tokenizer.")
    parser.add_argument("--n_instances", type=int, help="Number of test instances to evaluate.")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--gptq", action="store_true")
    
    args = parser.parse_args()
    if not args.model_name_or_path and not args.openai_engine:
        raise ValueError("Either --model_name_or_path or --openai_engine must be specified.")
    main(args)