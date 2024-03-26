#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# In[4]:


df = pd.read_csv("IMDB Dataset.csv")
df


# In[5]:


df.info()


# In[10]:


def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[\s+\.\!\/_,^%;:?()\"\'\|\[\]\-]+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply the cleaning function to the review column
df['review'] = df['review'].apply(clean_text)

# Display the cleaned dataframe
df.head()


# In[18]:


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords and initialize stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

# Define a function to tokenize, remove stopwords, and apply stemming
def process_text(text):
    # Tokenize the text to words
    words = word_tokenize(text)
    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    return words

# Apply the function to the review column
df['review_tokenized'] = df['review'].apply(process_text)

# Display the processed dataframe
df.head()


# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Join the tokens back into strings to prepare for vectorization
df['review_joined'] = df['review_tokenized'].apply(lambda tokens: ' '.join(tokens))

# Initialize the vectorizer with the desired number of features
tfidf_vectorizer = TfidfVectorizer(max_features=2000)

# Fit the vectorizer to the text data and transform it into TF-IDF features
X_tfidf = tfidf_vectorizer.fit_transform(df['review_joined'])

# Now `X_tfidf` is a sparse matrix representation of the reviews with TF-IDF values
X_tfidf


# In[27]:


# Converting categorical labels to numerical form

df["sentiment_numeric"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X_tfidf, df["sentiment_numeric"], test_size=0.2, random_state=1)


# In[28]:


# Scale data
scaler = MaxAbsScaler()

# Scale the training and test sets
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[29]:


# Train MLP model
mlp_model = MLPClassifier()
mlp_model.fit(x_train_scaled, y_train)


# In[30]:


# Evaluate the Model
y_pred = mlp_model.predict(x_test_scaled)

# Calculate accuracy
mlp_accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {mlp_accuracy}")

# Print classification report for a detailed performance analysis
print(classification_report(y_test, y_pred))


# In[33]:


# Define a function to get all scores
def get_scores(y_true, y_pred_prob, threshold=0.5):

    y_pred = (y_pred_prob[:, 1] >= threshold).astype(int)
    
    scores = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Roc_Auc': roc_auc_score(y_true, y_pred_prob[:, 1])
    }
    
    return scores

y_pred_mlp_prob = mlp_model.predict_proba(x_test_scaled)

# Now call the function with the true labels and the predicted probabilities
scores_mlp = get_scores(y_test, y_pred_mlp_prob)
scores_mlp


# In[ ]:




