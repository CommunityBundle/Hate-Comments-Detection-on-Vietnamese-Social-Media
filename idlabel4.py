import random
import string
import csv

def generate_unique_ids(prefix, length, count):
    if count > 26**length:
        raise ValueError("Cannot generate the requested number of unique IDs with the given length.")
    ids = set()
    while len(ids) < count:
        unique_part = ''.join(random.choices(string.ascii_lowercase, k=length))
        ids.add(f"{prefix}_{unique_part}")
    return list(ids)

def read_comments_from_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

# Parameters
prefix = "test"
id_length = 10

try:
    comments = read_comments_from_file("NLP-ML/VLSP2019-SHARED-Task-Hate-Speech-Detection-on-Social-Networks-Using-Bi-Lstm-master/Data/testdata.txt")
    unique_ids = generate_unique_ids(prefix, id_length, len(comments))

    with open("comments_for_test.csv", "w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # Add a comma before the header row
        file.write(",\n")
        writer.writerow(["order", "id", "free_text", "CLEAN", "OFFENSIVE", "HATE", "label_id"])
        
        # Write the data rows with order number starting from 0
        for order, (unique_id, comment) in enumerate(zip(unique_ids, comments)):
            writer.writerow([order, unique_id, f"{comment}", "", "", "", ""])
    
    print("Comments with IDs and order numbers have been written to 'comments_for_test'.")
except Exception as e:
    print(f"Error: {e}")
