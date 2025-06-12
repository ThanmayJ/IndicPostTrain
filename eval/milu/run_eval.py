import argparse
import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from eval.utils import get_next_word_predictions, load_hf_tokenizer, load_hf_lm, query_openai_chat_model, dynamic_import_function, upload_results_to_hf, check_and_upload_model_metadata


choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, batch_size=1):
    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    
    for i in tqdm(range(0, test_df.shape[0]), desc=f"Evaluating {subject}"):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "The answer is:"
            else:
                prompt += " The answer is:"

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        while len(tokenized_prompt) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is:"
                else:
                    prompt += " The answer is:"
            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        prompts.append(prompt)

    answer_choice_ids = [tokenizer.encode(" " + answer_choice, add_special_tokens=False)[-1] for answer_choice in choices]
    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, return_token_predictions=False, batch_size=batch_size
    )

    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)
        
    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    return cors, acc, all_probs


def eval_openai_chat_engine(args, subject, engine, dev_df, test_df, batch_size=1):
    import tiktoken
    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    answer_choice_ids = [gpt_tokenizer.encode(" " + x)[0] for x in choices]

    prompts = []
    for i in tqdm(range(0, test_df.shape[0]), desc=f"Evaluating {subject}"):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        prompts.append(prompt)

    instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=os.path.join(args.save_dir, f"{subject}_openai_results.jsonl"),
        logit_bias={token_id: 100 for token_id in answer_choice_ids},
        max_tokens=1,
    )
    
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(test_df)):
        prediction = results[i]["output"].strip()
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)
        
    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array([[0.25, 0.25, 0.25, 0.25] for _ in range(len(test_df))])
    return cors, acc, all_probs

def main(args):
    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        tokenizer = load_hf_tokenizer(model_name_or_path=args.model_name_or_path, revision=args.hf_revision, tokenizer_name_or_path=args.tokenizer_name_or_path, use_fast_tokenizer=not args.use_slow_tokenizer)
        model = load_hf_lm(model_name_or_path=args.model_name_or_path, revision=args.hf_revision, load_in_8bit=args.load_in_8bit, device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto", gptq_model=args.gptq)
        from transformers import GPTNeoXForCausalLM, OPTForCausalLM
        if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
            tokenizer.model_max_length = model.config.max_position_embeddings
            print(f"Set tokenizer.model_max_length to {model.config.max_position_embeddings}")
    
    subject = args.subject_name
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dev_csv_path = os.path.join(args.data_dir, args.lang if args.script=="native" else args.lang+"_"+args.script, "dev", f"{subject}_dev.csv")
    test_csv_path = os.path.join(args.data_dir, args.lang if args.script=="native" else args.lang+"_"+args.script, "test", f"{subject}_test.csv")


    if not os.path.isfile(dev_csv_path):
        raise FileNotFoundError(f"Dev file not found at: {dev_csv_path}")
    if not os.path.isfile(test_csv_path):
        raise FileNotFoundError(f"Test file not found at: {test_csv_path}")

    dev_df = pd.read_csv(dev_csv_path, header=None)[:args.ntrain]
    test_df = pd.read_csv(test_csv_path, header=None)
    
    if args.n_instances and args.n_instances < test_df.shape[0]:
        test_df = test_df.sample(args.n_instances, random_state=42)

    if args.model_name_or_path:
        cors, acc, probs = eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, args.eval_batch_size)
    else:
        cors, acc, probs = eval_openai_chat_engine(args, subject, args.openai_engine, dev_df, test_df, args.eval_batch_size)
        
    print("\n--- Final Result ---")
    print(f"Average accuracy for {subject}: {acc:.3f}")

    test_df["correct"] = cors
    for j in range(probs.shape[1]):
        choice = choices[j]
        test_df[f"choice{choice}_probs"] = probs[:, j]
    test_df.to_csv(os.path.join(args.save_dir, f"{subject}.csv"), index=None)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump({"average_accuracy": acc, "subject": subject}, f)

    if args.upload_to_hf is not None:
        results = {"average_accuracy": acc}
        task_name = f"oi_milu_{subject}_{args.ntrain}shots"
        upload_results_to_hf(results, args.upload_to_hf, args.hf_upload_name, task_name=task_name, primary_score=acc)
        check_and_upload_model_metadata(args.model_name_or_path, args.upload_to_hf, args.hf_upload_name, hf_revision=args.hf_revision)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data/eval/milu/", help="Directory where lang folders are present inside which dev/ and test/ folders are located.")
    parser.add_argument("--save_dir", type=str, default="results/milu/")
    parser.add_argument("--subject_name", type=str, default="general_knowledge", help="The base name of your CSV file pair (e.g., 'my_topic' for my_topic_dev.csv).")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--hf_revision", type=str, default=None, help="if specified, we will load the model from a revision of the model in the hub")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="if specified, we will load the tokenizer from here.")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If given, we will use the slow tokenizer.")
    parser.add_argument("--openai_engine", type=str, default=None, help="if specified, we will use the OpenAI API to generate the predictions.")
    parser.add_argument("--n_instances", type=int, help="if specified, a maximum of n_instances per subject will be used for the evaluation.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_chat_format", action="store_true", help="If given, we will use the chat format for the prompts.")
    parser.add_argument("--chat_formatting_function", type=str, default="eval.templates.create_prompt_with_tulu_chat_format", help="The function to use to create the chat format.")
    parser.add_argument("--upload_to_hf", type=str, default=None, help="If specified, we will upload the results to Hugging Face Datasets.")
    parser.add_argument("--hf_upload_name", type=str, default=None, help="If uploading to hf, this is the model name")
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        choices=["hi", "bn", "kn", "ml", "mr", "or", "pa", "ta", "te", "en"],
        help="Which language to evaluate on"
    )
    parser.add_argument(
        "--script",
        type=str,
        default="native",
        choices=["native", "roman"],
        help="Which script to evaluate on"
    )

    args = parser.parse_args()
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)