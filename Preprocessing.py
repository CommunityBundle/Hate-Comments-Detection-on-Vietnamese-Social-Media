import pandas as pd
import re
from pyvi import ViTokenizer

def preprocess_text(text):
    """
    Preprocess a given text according to the specified requirements.
    """
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove numerical characters
    text = re.sub(r'\d+', '', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    # Remove special characters but keep whitespace
    text = re.sub(r'[^\w\s]', '', text)
    
    # Add quotes around the cleaned text
    text = f'"{text}"'
    
    # Tokenize and normalize Vietnamese text
    text = ViTokenizer.tokenize(text)
    
    # Remove HTML, CSS, JS tags
    text = re.sub(r'<[^>]*>', '', text)
    
    return text

def preprocess_dataset(file_path, output_path):
    """
    Preprocess the dataset and save the processed version to a new file.
    """
    # Load the dataset
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Ensure the `free_text` column exists
    if 'free_text' not in df.columns:
        print("The dataset does not contain a 'free_text' column.")
        return
    
    # Apply preprocessing to the `free_text` column
    df['free_text'] = df['free_text'].apply(preprocess_text)
    
    # Save the processed dataset to a new file
    try:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Processed dataset saved to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

# Example usage
input_file = "VLSP2019-SHARED-Task-Hate-Speech-Detection-on-Social-Networks-Using-Bi-Lstm-master/Data/final_comments_train.csv"  # Replace with the path to your dataset
output_file = "VLSP2019-SHARED-Task-Hate-Speech-Detection-on-Social-Networks-Using-Bi-Lstm-master/Data/final_comments_train.csv"  # Replace with the desired output path
preprocess_dataset(input_file, output_file)
