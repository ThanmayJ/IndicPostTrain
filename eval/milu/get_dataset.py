import pandas as pd
from datasets import load_dataset # Install: pip install datasets pandas

def convert_index_to_letter(index):
    """Converts a 0-indexed integer to its corresponding uppercase letter (A=0, B=1, etc.)."""
    if not isinstance(index, int) or index < 0 or index > 25: # Basic validation for A-Z
        return None # Or raise an error, depending on desired error handling
    return chr(ord('A') + index)

def transform_image_data_to_standard_format(image_data):
    """
    Transforms data from the image's column format to the standard
    {"question": "...", "choices": ["...", "...", "...", "..."], "answer": 0} format.

    Args:
        image_data (list): A list of dictionaries, where each dictionary represents
                           a row from the image's structure (e.g., {'question': '...',
                           'option1': '...', 'option2': '...', 'option3': '...',
                           'option4': '...', 'target': 'optionX'}).

    Returns:
        list: A list of dictionaries in the standard format.
    """
    transformed_data = []
    option_to_index = {
        "option1": 0,
        "option2": 1,
        "option3": 2,
        "option4": 3
    }

    for item in image_data:
        question = item.get("question")
        option1 = item.get("option1")
        option2 = item.get("option2")
        option3 = item.get("option3")
        option4 = item.get("option4")
        target_option_str = item.get("target")

        if question is None or target_option_str is None:
            print(f"Warning: Skipping row due to missing 'question' or 'target': {item}")
            continue

        choices = [option1, option2, option3, option4]
        answer_index = option_to_index.get(target_option_str)

        if answer_index is None:
            print(f"Warning: Unknown target option '{target_option_str}' for question: '{question}'. Skipping row.")
            continue

        transformed_data.append({
            "question": question,
            "choices": choices,
            "answer": answer_index
        })
    return transformed_data

def process_and_save_dataset(data, output_csv_file="output_dataset.csv"):
    """
    Processes a list of dictionaries (in {"question": ..., "choices": [...], "answer": 0} format)
    into a pandas DataFrame and saves it to a CSV file.

    Args:
        data (list): A list of dictionaries, where each dictionary has 'question',
                     'choices' (list of 4 strings), and 'answer' (0-indexed int).
        output_csv_file (str): The name of the CSV file to save the data to.
    """
    processed_rows = []
    for item in data:
        question = item["question"]
        choices = item["choices"]
        answer_index = item["answer"]

        # Ensure we have exactly 4 choices, padding with empty strings if fewer
        # or truncating if more than 4, based on your desired CSV output.
        if len(choices) != 4:
            print(f"Warning: Choices list for question '{question}' does not have 4 elements. Adjusting.")
            # Pad with empty strings if less than 4
            padded_choices = choices + [''] * (4 - len(choices))
            # Truncate if more than 4
            final_choices = padded_choices[:4]
        else:
            final_choices = choices

        # Convert the answer index to a letter (e.g., 1 -> 'B')
        answer_letter = convert_index_to_letter(answer_index)
        if answer_letter is None:
            print(f"Warning: Invalid answer index {answer_index} for question '{question}'. Skipping row.")
            continue

        # Create a row for the DataFrame
        row = [question] + final_choices + [answer_letter]
        processed_rows.append(row)

    # Define column names for the CSV file
    csv_columns = ["question", "choice1", "choice2", "choice3", "choice4", "answer"]

    # Create a Pandas DataFrame
    df = pd.DataFrame(processed_rows, columns=csv_columns)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_file, index=False, header=False) # index=False prevents writing DataFrame index as a column

    print(f"Dataset successfully processed and saved to '{output_csv_file}'")
    # print("\nFirst few rows of the generated CSV:")
    # print(df.head())


id2lang = {
  "bn": "Bengali",
  "en": "English",
  "gu": "Gujarati",
  "hi": "Hindi",
  "kn": "Kannada",
  "ml": "Malayalam",
  "mr": "Marathi",
  "or": "Odia",
  "pa": "Punjabi",
  "ta": "Tamil",
  "te": "Telugu"
}
# --- Main execution block ---
if __name__ == "__main__":
    from sys import argv
    lang = argv[1]
    test_data = load_dataset("ai4bharat/MILU", id2lang[lang], split="test")
    dev_data = load_dataset("ai4bharat/MILU", id2lang[lang], split="validation")

    # Transform the data from the image's format to the standard format
    standard_formatted_test_data = transform_image_data_to_standard_format(test_data)
    standard_formatted_dev_data = transform_image_data_to_standard_format(dev_data)

    # Call the main processing function
    from os import makedirs
    test_path=f"data/eval/milu/{lang}/test"
    dev_path=f"data/eval/milu/{lang}/dev"
    makedirs(test_path, exist_ok=True)
    makedirs(dev_path, exist_ok=True)
    process_and_save_dataset(standard_formatted_test_data, f"{test_path}/general_knowledge_test.csv")
    process_and_save_dataset(standard_formatted_dev_data, f"{dev_path}/general_knowledge_dev.csv")
