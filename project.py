import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
#This library is used for creating visualizations, such as line plots and bar charts. It is used to plot the graphs showing the actual vs. predicted stress levels.
import matplotlib.pyplot as plt 

# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Read the dataset from CSV file
dataset = pd.read_csv('Emotion_classify_Data.csv')

# Text preprocessing steps
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove special characters and numbers(any thing not az)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # Tokenization and lowercase conversion
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and perform stemming and lemmatization
    tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens if token not in stop_words]
    
    # Join tokens back into a single string
    processed_text = ' '.join(tokens)
    
    return processed_text

# Apply preprocessing to the 'text' column
dataset['processed_text'] = dataset['Comment'].apply(preprocess_text)

# TF-IDF feature extraction
tfidf_vectorizer = TfidfVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(dataset['processed_text'])
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# Bag of Words feature extraction
bow_vectorizer = CountVectorizer()
bow_features = bow_vectorizer.fit_transform(dataset['processed_text'])
bow_feature_names = bow_vectorizer.get_feature_names()

# POS tags feature extraction
pos_tags = []
for text in dataset['processed_text']:
    tokens = word_tokenize(text)
    pos_tags.append(nltk.pos_tag(tokens))

# Create output directories if they don't exist
os.makedirs('output', exist_ok=True)

# Save the preprocessed dataset to a new CSV file
preprocessed_dataset_path = os.path.join('output', 'preprocessed_dataset.csv')
dataset.to_csv(preprocessed_dataset_path, index=False)

# Save the extracted features to separate text files
tfidf_features_path = os.path.join('output', 'tfidf_features.txt')
with open(tfidf_features_path, 'w') as file:
    file.write(str(tfidf_features.toarray()))
    
bow_features_path = os.path.join('output', 'bow_features.txt')
with open(bow_features_path, 'w') as file:
    file.write(str(bow_features.toarray()))
    
pos_tags_path = os.path.join('output', 'pos_tags.txt')
with open(pos_tags_path, 'w') as file:
    for tags in pos_tags:
        file.write(str(tags) + '\n')

# Combine the features with the original dataset
combined_features = pd.concat([dataset, pd.DataFrame(tfidf_features.toarray(), columns=tfidf_feature_names), pd.DataFrame(bow_features.toarray(), columns=bow_feature_names)], axis=1)
# Save the combined dataset to a new CSV file
combined_features_path = os.path.join('output', 'combined_features.csv')
combined_features.to_csv(combined_features_path, index=False)

# Print the combined dataset
print("Combined dataset:")
print(combined_features)

# Print a confirmation message
print("Preprocessed dataset saved to", preprocessed_dataset_path)
print("TF-IDF features saved to", tfidf_features_path)
print("Bag of Words features saved to", bow_features_path)
print("POS tags saved to", pos_tags_path)


########################## for asking user not sure edite later #####################

# Train a classifier (Logistic Regression) using the TF-IDF features
classifier = LogisticRegression()
classifier.fit(tfidf_features, dataset['Class'])

# Classify user input
user_input = input("Enter a sentence: ")

# Preprocess user input
processed_user_input = preprocess_text(user_input)

# Convert preprocessed user input to TF-IDF features
user_input_features = tfidf_vectorizer.transform([processed_user_input])

# Classify user input using the trained classifier
predicted_class = classifier.predict(user_input_features)[0]

# Get the class label based on the predicted class index
predicted_class_label = dataset['Class'].unique()[predicted_class]

# Print the predicted class label
print("Predicted class:", predicted_class_label)
