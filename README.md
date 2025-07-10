# Movie Sentiment Analyzer

This project analyzes IMDB movie reviews and classifies their **sentiment** (positive or negative) using **Natural Language Processing (NLP)** techniques and a **Multi-Layer Perceptron (MLP)** classifier. It covers everything from data preprocessing to model evaluation, and demonstrates strong performance on a large-scale sentiment classification task.

### Features
- ✅ Developed a sentiment analysis model achieving **87.62% accuracy** using TF-IDF vectorization and an **MLP classifier**
- 🔧 Optimized text preprocessing with **NLTK** (tokenization, stopword removal, stemming), improving feature quality and boosting classification performance by **15%**
- 📊 Evaluated model using:
  - **Precision**: 86.96%
  - **Recall**: 88.26%
  - **F1 Score**: 87.60%
  - **ROC-AUC**: 94.84%

### Technologies Used

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
| **Notebook Interface** | Jupyter                                                                |
| **Data Source**        | Kaggle (IMDB Sentiment Dataset)                                        |

(line10.png)
