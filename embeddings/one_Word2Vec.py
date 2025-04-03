import kagglehub, os
import pandas as pd
import re
import nltk 
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.data.path.append("C:/Users/a.khan/AppData/Roaming/nltk_data") # NLTK data ka path day rahay hain yahan hum

# Download necessary NLTK data files
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

dataset_path = r"C:\Users\PC\.cache\kagglehub\datasets\vishakhdapat\imdb-movie-reviews\versions\1"
print("files of downloaded dataset : ", os.listdir(dataset_path))

file_path = os.path.join(dataset_path, "IMDB Dataset.csv")
df = pd.read_csv(file_path)

print("Displaying first few rows", df.head())
print(df.info())  # Column names, data types
print(df.describe())  # Summary stats for numerical data
print(df['sentiment'].value_counts())  # Count of positive/negative reviews

lemmatizer = WordNetLemmatizer() # Initialize lemmatizer and stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & stopword removal
    return words  # Returning list of words instead of string (needed for Word2Vec) e.g. sentences = [['this', 'is', 'a', 'great', 'movie'], ['i', 'love', 'movies']]

df['tokenized_review'] = df['review'].apply(clean_text) # tokennized means sentences = [['this', 'is', 'a', 'great', 'movie'], ['i', 'love', 'movies']]
print(df[['review', 'tokenized_review']].head())

############################################################################################################################################
##########################1st argumnet : df['tokenized_review'] contains tokenized (preprocessed) reviews. ['one', 'reviewer','episode'], ...
##########################2nd argumnet : vector_size=10 here Each word in the vocabulary is represented by a 10-dimensional numerical vector 
################################### print(word2vec_model.wv['great']) => [ 0.1458  0.8721 -0.3845 0.8721 -0.3845 0.8721 -0.3845 0.2313 -0.1121  0.5067]
##########################3rd argument : window=3 here it looks 3 words before and after a target word to understand its context
##########################4th argument : min_count=2 Ignores words that appear less than twice in the dataset to reduce noise
##########################5th argument : workers=4 Uses 4 CPU cores to train the model faster (parallel processing).
word2vec_model = Word2Vec(sentences=df['tokenized_review'], vector_size=10, window=3, min_count=2, workers=4) # Train Word2Vec Model
word2vec_model.save("word2vec.model")
print("Word2Vec model trained and saved.")

# Function to convert review to word vector (by averaging word embeddings)
def get_review_vector(words, model, vector_size):
    vector = np.zeros(vector_size)  # Initialize zero vector
    valid_words = 0  # Count of valid words found in Word2Vec
    for word in words:
        if word in model.wv:
            vector += model.wv[word]
            valid_words += 1
    if valid_words > 0:
        vector /= valid_words  # Averaging : Makes all reviews have the same vector size, Balances short and long reviews so that longer reviews don't get unfairly large values
                               # Preserves semantic meaning (similar reviews will have similar vectors)
    return vector

# Convert all reviews into word vectors
vector_size = 10  # Same as Word2Vec vector_size
X = np.array([get_review_vector(words, word2vec_model, vector_size) for words in df['tokenized_review']])
#multiline equivalnet of above 
# Initialize an empty list to store review vectors
# review_vectors = []
# # Loop through each review (which is a list of tokenized words)
# for words in df['tokenized_review']:
#     # Convert the list of words into a numerical vector using Word2Vec
#     vector = get_review_vector(words, word2vec_model, vector_size)
#     # Append the generated vector to the review_vectors list
#     review_vectors.append(vector)
# # Convert the list of review vectors into a NumPy array
# X = np.array(review_vectors)

# Convert sentiment labels to numerical values
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
y = df['sentiment']

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training Set Size: {X_train.shape}, Testing Set Size: {X_test.shape}")

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Word2Vec Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))




####################################################################################################
####SIMPLE EXAMPLE SIMPLE EXAMPLESIMPLE EXAMPLE#####################################################
####SIMPLE EXAMPLE SIMPLE EXAMPLESIMPLE EXAMPLE#####################################################
# Let's say Word2Vec learned these vectors:
# "good"  → [0.1, 0.3, 0.5]
# "movie" → [0.2, 0.4, 0.6]
# "boring" → [-0.3, -0.2, -0.1]

# Input Review:
# words = ["good", "movie", "boring"]

# Processing:
# vector = np.zeros(3)  # [0.0, 0.0, 0.0]
# vector += [0.1, 0.3, 0.5]  # good
# vector += [0.2, 0.4, 0.6]  # movie
# vector += [-0.3, -0.2, -0.1]  # boring

# After Sum:
# vector = [0.0, 0.5, 1.0]

# Now, divide by 3:
# vector = [0.0, 0.166, 0.333]
#########################################################################################################################################
#########################################################################################################################################