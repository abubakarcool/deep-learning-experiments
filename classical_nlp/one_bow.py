import kagglehub,os
import pandas as pd
import re
import nltk 
nltk.data.path.append("C:/Users/a.khan/AppData/Roaming/nltk_data")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# path = kagglehub.dataset_download("vishakhdapat/imdb-movie-reviews") # Download latest version
# print("Path to dataset files:", path)
dataset_path = r"C:\Users\a.khan\.cache\kagglehub\datasets\vishakhdapat\imdb-movie-reviews\versions\1"
print("files of downloaded dataset : ",os.listdir(dataset_path))

file_path = os.path.join(dataset_path, "IMDB Dataset.csv") 
df = pd.read_csv(file_path)

print("Displaying first few rows",df.head())

print(df.info())  # Column names, data types
print(df.describe())  # Summary stats for numerical data
print(df['sentiment'].value_counts())  # Replace 'column_name' with actual label column name



# Step 1: Data Cleaning and Preprocessing
# Since our dataset contains text reviews, we need to clean and preprocess the text to prepare it for the BoW model.
# 1. Remove HTML tags (some reviews have <br /> tags).
# 2. Convert text to lowercase (BoW is case-sensitive).
# 3. Remove special characters, numbers, and punctuation.
# 4. Tokenization (split text into words).
# 5. Remove stopwords (common words like "the", "and", "is").
# 6. Lemmatization (convert words to their base form, e.g., "running" → "run").


# Download necessary NLTK data files
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) # Load English stopwords

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & stopword removal
    return ' '.join(words)

df['cleaned_review'] = df['review'].apply(clean_text)
print(df[['review', 'cleaned_review']].head())


# Step 2: Convert Text into Bag-of-Words (BoW) Representation
# Now, we need to transform the cleaned text into a numerical format using the CountVectorizer.
# 1. Convert cleaned text into a matrix of token counts (BoW representation).
# 2. Display the shape of the transformed data.
# 3. Check the vocabulary size.

vectorizer = CountVectorizer(max_features=500)  # Limit vocabulary to 500 most frequent words
X = vectorizer.fit_transform(df['cleaned_review'])

bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()) # Convert to DataFrame
print(f"BoW Matrix Shape: {bow_df.shape}")  # (50000, 500) # Display shape and first few rows
print(bow_df.head())
print(f"Vocabulary Size: {len(vectorizer.get_feature_names_out())}") # Display vocabulary size



# Step 3: Train a Machine Learning Model on BoW Data
# We will now use the BoW features to train a simple Logistic Regression model to classify reviews as positive or negative
# 1. Convert the sentiment labels into numerical values (positive = 1, negative = 0).
# 2. Split the dataset into training (80%) and testing (20%) sets.
# 3. Train a Logistic Regression classifier.
# 4. Evaluate the model on the test set.

df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0}) # Convert sentiment labels to numeric values

X = bow_df #ipnut for the model
y = df['sentiment'] #output or result of the model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split dataset into 80% training and 20% testing

print(f"Training Set Size: {X_train.shape}, Testing Set Size: {X_test.shape}")

model = LogisticRegression(max_iter=1000) # Initialize the Logistic Regression model
model.fit(X_train, y_train) # Train the model
y_pred = model.predict(X_test) # Predict on test data

accuracy = accuracy_score(y_test, y_pred) # Evaluate model performance
print(f"Model Accuracy: {accuracy:.4f}")

print("Classification Report:\n", classification_report(y_test, y_pred)) # Print classification report