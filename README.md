# Movie Sentiment Analyzer

This project analyzes IMDB movie reviews and classifies their **sentiment** (positive or negative) using **Natural Language Processing (NLP)** and a **Multi-Layer Perceptron (MLP)** model. It includes full preprocessing, vectorization, model training, and evaluation using modern machine learning tools.

### 🧠 Features
- ✅ Cleaned and preprocessed raw text reviews using **regex**, **NLTK stopwords**, and **stemming**
- 🧹 Tokenized reviews and applied **Snowball stemming**
- ✨ Transformed text data into TF-IDF vectors (**2000 features**) for model input
- 🤖 Built and trained an **MLPClassifier** using **scikit-learn**
- 📊 Evaluated model with metrics including **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC**

### 🛠️ Technologies Used

| Category               | Tools & Libraries                                                      |
|------------------------|------------------------------------------------------------------------|
| **Language**           | Python                                                                 |
| **Data Handling**      | pandas                                                                 |
| **Text Cleaning/NLP**  | re, NLTK (stopwords, tokenization, stemming)                           |
| **Vectorization**      | scikit-learn `TfidfVectorizer`                                         |
| **Scaling**            | scikit-learn `MaxAbsScaler`                                            |
| **Modeling**           | scikit-learn `MLPClassifier`                                           |
| **Evaluation**         | scikit-learn metrics (`accuracy_score`, `precision`, `recall`, etc.)   |
| **Visualization**      | matplotlib, seaborn (optional setup for EDA and plotting)              |

