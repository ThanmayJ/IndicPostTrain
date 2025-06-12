import pandas as pd
from datasets import load_dataset

def convert_index_to_letter(index):
    """Converts a 0-indexed integer to its corresponding uppercase letter (A=0, B=1, etc.)."""
    if not isinstance(index, int) or index < 0 or index > 25: # Basic validation for A-Z
        return None # Or raise an error
    return chr(ord('A') + index)

def process_and_save_dataset(data, output_csv_file="output_dataset.csv"):
    """
    Processes a list of dictionaries into a pandas DataFrame and saves it to a CSV file.

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

        # Create a row for the DataFrame
        row = [question] + final_choices + [answer_letter]
        processed_rows.append(row)

    # Define column names for the CSV file
    csv_columns = ["question", "choice1", "choice2", "choice3", "choice4", "answer"]

    # Create a Pandas DataFrame
    df = pd.DataFrame(processed_rows, columns=csv_columns)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_file, header=False, index=False) # index=False prevents writing DataFrame index as a column

    print(f"Dataset successfully processed and saved to '{output_csv_file}'")
    # print("\nFirst few rows of the generated CSV:")
    # print(df.head())

# --- Main execution block ---
if __name__ == "__main__":
    from sys import argv
    lang = argv[1]
    test_data = load_dataset("sarvamai/mmlu-indic", lang, split="test")
    dev_data = load_dataset("sarvamai/mmlu-indic", lang, split="validation")

    # Call the main processing function
    from os import makedirs
    test_path=f"data/eval/mmlu-indic/{lang}/test"
    dev_path=f"data/eval/mmlu-indic/{lang}/dev"
    makedirs(test_path, exist_ok=True)
    makedirs(dev_path, exist_ok=True)
    process_and_save_dataset(test_data, f"{test_path}/general_knowledge_test.csv")
    process_and_save_dataset(dev_data, f"{dev_path}/general_knowledge_dev.csv")