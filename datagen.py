import joblib
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
nltk.download('punkt')
nltk.download('stopwords')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import gensim
import os
import requests
from tqdm import tqdm
import zipfile
from save_load import save
from sklearn.preprocessing import LabelEncoder
from nltk import bigrams
from collections import Counter


# Preprocessing function
def preprocess_text(text):

    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    # Tokenization
    tokens = word_tokenize(text)

    # Lowercasing
    tokens = [word.lower() for word in tokens]

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens)


def download_file(url, save_path):
    # Streaming download with progress bar using requests
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    else:
        print("Download successful")


def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


# Function to convert tweet to GloVe vector
def tweet_to_glove(tweet, model):
    words = tweet.split()
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)


def datagen():

    data = pd.read_csv('C:\\Users\\hardi\\OneDrive\\Desktop\\Research\\Ph.D_Python\\Python code\\Client-Mrs.Tanci\\final_data.csv')

    label = data['class']

    # HEXACO
    # H - Honesty - Humility,
    # E - Emotionality,
    # X - Extra version,
    # A - Agreeableness,
    # C - Conscientiousness
    # O - Openness to Experience

    # Count the occurrences of each label
    label_counts = label.value_counts()

    # Define full forms of the HEXACO traits
    full_forms = {
        'H': 'Honesty-Humility',
        'E': 'Emotionality',
        'X': 'Extraversion',
        'A': 'Agreeableness',
        'C': 'Conscientiousness',
        'O': 'Openness to Experience'
    }

    # Map the labels to their full forms
    labels_full_form = [full_forms[key] for key in label_counts.index]

    # Create pie chart
    plt.figure(figsize=(12, 10))
    explode = [0.02] * len(label_counts)
    plt.pie(label_counts, labels=[''] * len(label_counts), startangle=140, explode=explode)
    plt.axis('equal')
    plt.legend(labels_full_form, loc='upper right', prop={'size': 11})
    plt.title('Distribution of Labels')
    plt.savefig('Data Visualization/label.png')
    plt.show()

    all_text = ' '.join(data['text'][:500])

    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_text)

    # Display the word cloud
    plt.figure(figsize=(14, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Tweets')
    plt.savefig('./Data Visualization/Word Cloud of Tweets.png')
    plt.show()

    # Apply preprocessing
    data['preprocessed tweet'] = data['text'].apply(preprocess_text)

    # TF-IDF Vectorization with n-grams
    tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))  # bi-grams included
    X_tfidf = tfidf.fit_transform(data['preprocessed tweet']).toarray()

    # URL to download glove embeddings
    if 'glove.6B.zip' not in os.listdir('Saved Data'):
        url = 'https://nlp.stanford.edu/data/glove.6B.zip'

        # Directory where you want to save the downloaded file
        save_dir = 'Saved Data'

        # Ensure the directory exists, create if necessary
        os.makedirs(save_dir, exist_ok=True)

        # File path to save the downloaded zip file
        zip_file_path = os.path.join(save_dir, 'glove.6B.zip')

        # Download the file
        download_file(url, zip_file_path)

        extract_dir = 'Saved Data'
        extract_zip(zip_file_path, extract_dir)

    # GloVe model
    glove_file = 'Saved Data/glove.6B.100d.txt'  # Ensure this file is available in your directory
    glove_model = gensim.models.KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

    # Apply GloVe vectorization
    data['glove_vector'] = data['preprocessed tweet'].apply(lambda x: tweet_to_glove(x, glove_model))
    X_glove = np.stack(data['glove_vector'].values)

    features = np.concatenate([X_tfidf, X_glove], axis=1)

    features = abs(features)

    features = features/np.max(features, axis=0)

    features = np.nan_to_num(features)

    label = data['class']

    lab_encoder = LabelEncoder()
    labels = lab_encoder.fit_transform(label)
    joblib.dump(lab_encoder, "Saved Data/label encoder.joblib")

    train_sizes = [0.7, 0.8]
    for train_size in train_sizes:
        x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=train_size)
        save('x_train_' + str(int(train_size * 100)), x_train)
        save('y_train_' + str(int(train_size * 100)), y_train)
        save('x_test_' + str(int(train_size * 100)), x_test)
        save('y_test_' + str(int(train_size * 100)), y_test)
