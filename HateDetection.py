from keras._tf_keras.keras.models import model_from_json
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import pandas as pd  # Needed for DataFrame operations
import matplotlib as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score,auc,roc_curve,average_precision_score, precision_recall_curve, average_precision_score

# Load model architecture
with open('model_num_bc.json', "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
print("Model architecture loaded successfully.")

# Load weights into the model
model.load_weights("model.weights.h5")
print("Model weights loaded successfully.")
model.summary()

train = pd.read_csv("VLSP2019-SHARED-Task-Hate-Speech-Detection-on-Social-Networks-Using-Bi-Lstm-master/Data/final_comments_train.csv").fillna(" ")
# Assuming train_x is a Pandas DataFrame
label_counts = train['label_id'].value_counts()

# Extract the counts for each label
clean_count = label_counts.get(0, 0)  # Get count for 'CLEAN' (label_id = 0)
offensive_count = label_counts.get(1, 0)  # Get count for 'OFFENSIVE' (label_id = 1)
hate_count = label_counts.get(2, 0)  # Get count for 'HATE' (label_id = 2)

print(f"Total Clean Comments: {clean_count}")
print(f"Total Offensive Comments: {offensive_count}")
print(f"Total Hate Comments: {hate_count}")


# Load the tokenizer
with open("tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
print("Tokenizer loaded successfully.")
# Dataset Test
test = pd.read_csv("VLSP2019-SHARED-Task-Hate-Speech-Detection-on-Social-Networks-Using-Bi-Lstm-master/Data/comments_processed_test.csv").fillna(" ")
test = test['free_text']
test_seq = tokenizer.texts_to_sequences(test)
test_seq_padded = pad_sequences(test_seq,maxlen=150)
# print(f"Original dataset size: {len(test)}")  

# Input text to predict
input_text = "em gái mày nhìn ngu v? nó ngu sẵn từ lúc sinh ra r 🤓🤝🗿"
# print(f"Raw input: {input_text}")

# Preprocess the text
input_seq = tokenizer.texts_to_sequences([input_text])  # Tokenize text
input_seq_padded = pad_sequences(input_seq, maxlen=150)  # Pad sequence to match training

# Make prediction
predictions = model.predict(test_seq_padded)
# print(f"Predictions: {predictions}")
result = pd.read_csv('VLSP2019-SHARED-Task-Hate-Speech-Detection-on-Social-Networks-Using-Bi-Lstm-master/Data/comments_processed_test.csv')
result[['CLEAN', 'OFFENSIVE', 'HATE']] = predictions
for i in range(len(result)):
    if (result['CLEAN'][i] >= result['OFFENSIVE'][i] and result['CLEAN'][i]>=result['HATE'][i]):
        result['label_id'][i]=int(0)
    elif(result['OFFENSIVE'][i] >= result['CLEAN'][i] and result['OFFENSIVE'][i]>=result['HATE'][i]):
        result['label_id'][i] = int(1)
    elif (result['HATE'][i] >= result['OFFENSIVE'][i] and result['HATE'][i] >= result['CLEAN'][i]):
        result['label_id'][i] = int(2)
result.to_csv("Result_samples.csv")
# Convert predictions to a DataFrame for processing
result = pd.DataFrame(predictions, columns=['CLEAN', 'OFFENSIVE', 'HATE'])

# Assign label_id based on the highest probability
result['label_id'] = np.argmax(result.values, axis=1)

# Display the result

label_map = {0: 'CLEAN', 1: 'OFFENSIVE', 2: 'HATE'}
result['label'] = result['label_id'].map(label_map)
# Ensure all rows are displayed
# pd.set_option('display.max_rows', None)  # Set to None to display all rows
# pd.set_option('display.max_columns', None)  # Set to None to display all columns
# print(result)

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
test_results = pd.read_csv("Result_samples.csv")

# Extract true labels and predicted probabilities
y_true = test_results['label_id'].values
y_pred_proba = test_results[['CLEAN', 'OFFENSIVE', 'HATE']].values

# Convert true labels to one-hot encoding
num_classes = 3
y_true_one_hot = pd.get_dummies(y_true).values  # One-hot encode true labels

# Initialize storage for AUC scores and classification metrics
roc_auc_scores = []
prc_auc_scores = []

# Initialize storage for metrics
f1_scores = []
metrics_summary = []

# Confusion matrix per class
conf_matrices = []

# Evaluate metrics for each class
for i in range(num_classes):
    y_true_class = y_true_one_hot[:, i]
    y_pred_class = y_pred_proba[:, i]
    
    # Calculate Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true_class, y_pred_class)
    prc_auc = auc(recall, precision)
    prc_auc_scores.append(prc_auc)

    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Class {i} (PRC-AUC = {prc_auc:.3f})', color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for Class {i}')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()
    
    # Generate confusion matrix
    y_pred_labels = (y_pred_class > 0.5).astype(int)  # Binarize predictions
    tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_labels).ravel()
    conf_matrices.append((tn, fp, fn, tp))

    # Calculate F1-score
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    f1_scores.append(f1)

    # Store metrics summary
    metrics_summary.append({'Class': i, 'PRC-AUC': prc_auc, 'F1-Score': f1, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn})

def plot_confusion_matrices(conf_matrices):
    fig, axes = plt.subplots(1, len(conf_matrices), figsize=(15, 5))
    for i, cm in enumerate(conf_matrices):
        labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        counts = [f"{value}" for value in cm]
        percentages = [f"{value:.2%}" for value in cm / np.sum(cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(labels, counts, percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        
        sns.heatmap(np.array(cm).reshape(2, 2), annot=labels, fmt='', cmap='Blues', ax=axes[i])
        axes[i].set_xlabel('Predicted labels')
        axes[i].set_ylabel('True labels')
        axes[i].set_title(f'Confusion Matrix for Class {i}')
    plt.tight_layout()
    plt.show()
# Visualize confusion matrices
# plot_confusion_matrices(conf_matrices)

# Print metrics summary
for metric in metrics_summary:
    print(f"Class {metric['Class']} - PRC-AUC: {metric['PRC-AUC']:.3f}, F1-Score: {metric['F1-Score']:.3f}, "
          f"TP: {metric['TP']}, FP: {metric['FP']}, TN: {metric['TN']}, FN: {metric['FN']}")
