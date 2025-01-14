import numpy as np
import pandas as pd
import pickle
from keras._tf_keras.keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras._tf_keras.keras.layers import Dropout, Embedding
from keras._tf_keras.keras.preprocessing import text, sequence
from keras._tf_keras.keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras import backend as K
from keras._tf_keras.keras.models import model_from_json
from keras._tf_keras.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt



EMBEDDING_FILE = 'cc.vi.300.vec'
train_x = pd.read_csv('VLSP2019-SHARED-Task-Hate-Speech-Detection-on-Social-Networks-Using-Bi-Lstm-master/Data/final_comments_train.csv').fillna(" ")
test_x = pd.read_csv('VLSP2019-SHARED-Task-Hate-Speech-Detection-on-Social-Networks-Using-Bi-Lstm-master/Data/comments_processed_test.csv').fillna(" ")
# Load training and test datasets into Pandas DataFrames, replacing missing values with empty strings.
# print("Training data after loaded in Pandas DataFrames:\n",train_x.head())

#Basic visualization of data using histograms
# FacetGrid- Multi-plot grid for plotting conditional relationships

# train_x['text_length'] = train_x['free_text'].apply(len)
# graph = sns.FacetGrid(data=train_x, col='label_id ')
# graph.map(plt.hist, 'text_length', bins=50)
# plt.show()

max_features=2272
maxlen=150
embed_size=300

train_x['free_text'].fillna(' ')
# Fill missing values in the free_text column with spaces.
test_x['free_text'].fillna(' ')
train_y = train_x[['CLEAN','OFFENSIVE','HATE']].values

train_x = train_x['free_text'].str.lower()
test_x = test_x['free_text'].str.lower()
# Step 1: Convert to string type
train_y = train_y.astype(str)

# Step 2: Strip whitespace
train_y = np.char.strip(train_y)

# Step 3: Replace empty strings with NaN
train_y[train_y == ''] = np.nan

# Step 4: Convert to numeric, replacing errors with NaN
train_y = np.where(np.isnan(train_y.astype(float)), np.nan, train_y.astype(float))

# Step 5: Handle NaN values (fill with 0 or drop)
train_y = np.nan_to_num(train_y, nan=0)  # Change to np.delete to drop NaNs

# Check the cleaned data
print("Cleaned Train Y:", train_y)
# Vectorize text + Prepare  Embedding
tokenizer = text.Tokenizer(num_words=max_features, lower=True)
tokenizer.fit_on_texts(list(train_x))
word_index = tokenizer.word_index
# create a vocaulary for the dataset. NOTE:THE INDICES ARE SORTED ACCORDING TO WORD FREQUENCY.
print("Vocabulary of Training data:\n", list(word_index.items())[:10])

train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)
print("train data after we convert sentences into sequences of integer:\n",train_x[:10])
# Tokenization: Converts text into sequences of integer indices using the tokenizer.
# Each word in the text is replaced with its corresponding integer index from word_index

train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)
print("padding the sentences that didn't reach the maxlen:\n",train_x[:10])
# Sequences from different sentences may have varying lengths. To ensure uniform input for deep learning models, padding is applied.
print("create vector")
embeddings_index = {}
i = 0
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0] 
        
        coefs = np.asarray(values[1:], dtype='float32')
        # if(i < 10):
        #     print(word)
        #     print(coefs)
        #     i += 1
        embeddings_index[word] = coefs
        
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue

    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# Establish a word matrix using a pre-trained word embeddings.

# Dimensionality reduction using t-SNE / Visualization of Word Embeddings
# from sklearn.manifold import TSNE
# print("Performing t-SNE on the embedding matrix for visualization...")
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
# reduced_embeddings = tsne.fit_transform(embedding_matrix)

# # Plot the t-SNE result
# plt.figure(figsize=(12, 10))
# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=5, alpha=0.6)

# # Annotate points with words (optional, limit to top 50 words)
# word_to_index = {word: idx for word, idx in word_index.items() if idx < len(reduced_embeddings)}
# for word, idx in list(word_to_index.items())[:50]:  # Limit annotations to top 50 words
#     plt.annotate(word, (reduced_embeddings[idx, 0], reduced_embeddings[idx, 1]), fontsize=8)

plt.title("Word Embedding Visualization (t-SNE)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

# Build Model
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x = SpatialDropout1D(0.35)(x)

x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)


avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])

out = Dense(3, activation='sigmoid')(x)

model = Model(inp, out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Prediction
batch_size = 32
epochs = 3
model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1)
predictions = model.predict(test_x, batch_size=batch_size, verbose=1)

print(predictions)
print(predictions.shape)

result = pd.read_csv('VLSP2019-SHARED-Task-Hate-Speech-Detection-on-Social-Networks-Using-Bi-Lstm-master/Data/comments_processed_test.csv')
print(result.shape)
result[['CLEAN', 'OFFENSIVE', 'HATE']] = predictions
# submission.to_csv('Thunghiem1.csv', index=False)
model_json = model.to_json()
with open("model_num_bc.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.weights.h5")
for i in range(len(result)):
    if (result['CLEAN'][i] >= result['OFFENSIVE'][i] and result['CLEAN'][i]>=result['HATE'][i]):
        result['label_id'][i]=int(0)
    elif(result['OFFENSIVE'][i] >= result['CLEAN'][i] and result['OFFENSIVE'][i]>=result['HATE'][i]):
        result['label_id'][i] = int(1)
    elif (result['HATE'][i] >= result['OFFENSIVE'][i] and result['HATE'][i] >= result['CLEAN'][i]):
        result['label_id'][i] = int(2)


# Save the tokenizer to a file
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

print("Tokenizer saved successfully.")


result.to_csv("Result_latest.csv")





