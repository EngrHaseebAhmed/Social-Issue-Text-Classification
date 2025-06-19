# 🗂️ Reddit Social Issues Dataset

This dataset captures Reddit user discussions related to social infrastructure issues in Asian Cities, focusing on topics like electricity, water, waste management, traffic, and security. It includes both a **cleaned CSV** version and a **tokenized JSON** format ready for machine learning.

---

## 📁 Files Included
| File | Description |
|------|-------------|
| `reddit_social_issues_cleaned.csv` | Cleaned, labeled dataset with structured Reddit posts and metadata |
| `preprocessed_reddit_social_issues_cleaned.json` | Tokenized version of the dataset, ready for transformer-based NLP models (e.g., BERT) |
| `Social_Issues_Classification.ipynb` | Social_Issues_Classification.ipynb): Notebook for preprocessing and classifying social issues |


---

## 🧾 Dataset Overview

Each Reddit post has been preprocessed and labeled under one of the following issue categories:
- Electricity
- Water
- Waste Management
- Traffic
- Security

### CSV File (`reddit_social_issues_cleaned.csv`)

| Column | Description |
|--------|-------------|
| `issue` | Assigned category label |
| `subreddit` | Source subreddit |
| `title` | Post title |
| `post_text` | Body of the post |
| `created_utc` | Timestamp |
| `url` | Source URL |
| `keyword` | Extracted keyword used in labeling |

### JSON File (`preprocessed_reddit_social_issues_cleaned.json`)

Each entry is a dictionary with:
```json
{
  "text": "tokenized or normalized post text",
  "label": "Electricity",
  "sentiment": 0.12,
  "urgent": false
}
```

| Field       | Description                                    |
| ----------- | ---------------------------------------------- |
| `text`      | Preprocessed and tokenized version of the post |
| `label`     | Labeled issue category                         |
| `sentiment` | Sentiment score (float, from -1 to 1)          |
| `urgent`    | Boolean flag indicating urgency                |

---

## 🧼 Cleaning and Preprocessing

- **Keyword-Based Filtering**: Custom include/exclude regex rules for each category to remove mislabeled posts.
- **Semantic Filtering** *(optional)*: Used Sentence Transformers for cosine similarity relevance scoring.
- **Tokenization**: Applied lowercasing, stopword removal, and encoding for model training.

---
## 📊 Project Implementation: Social Issues Classification using NLP

This project implements a full NLP pipeline to classify Reddit posts based on different social issues such as crime, electricity, waste management, harassment, and more. It showcases modern data preprocessing, feature engineering, and classical machine learning applied to real-world text data.

### 🚀 Objectives
- Classify Reddit posts into relevant social issue categories.
- Apply NLP preprocessing and sentiment analysis to raw text data.
- Train and evaluate machine learning models on the cleaned dataset.

---
## 📁 Project Structure

### 🛠️ Implementation Details

#### 1. **Data Collection**
- The dataset comprises Reddit posts related to various societal issues.
- Each post includes a title, body text, and a manually labeled category.

#### 2. **Text Preprocessing**
- Merged the `title` and `text` columns for unified input.
- Cleaned the text using regex to remove:
  - URLs
  - Numbers
  - HTML tags
  - Punctuation and special characters
- Converted text to lowercase and normalized whitespace.
- Removed stopwords using SpaCy.
- Applied lemmatization using SpaCy's `en_core_web_sm` model.
- Generated sentiment polarity scores using TextBlob.

#### 3. **Exploratory Data Analysis (EDA)**
- Analyzed word count and class distribution.
- Visualized insights using `matplotlib` and `seaborn`.

#### 4. **Feature Engineering**
- Transformed text into numerical features using **TF-IDF Vectorization**.
- Created a sparse matrix representing weighted word frequencies.

#### 5. **Model Training**
- Evaluated multiple classical ML models:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
- Used `GridSearchCV` for hyperparameter tuning and model optimization.

#### 6. **Model Evaluation**
- Measured performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Displayed confusion matrices for detailed class-wise analysis.

---

### 🧠 Key NLP Techniques
- Tokenization and Lemmatization (SpaCy)
- Sentiment Analysis (TextBlob)
- TF-IDF Vectorization
- Multi-class Text Classification
- Evaluation and Visualization

---




### Load CSV (Pandas)
```python
import pandas as pd
df = pd.read_csv("reddit_social_issues_cleaned.csv")
print(df.head())
```

### Load Tokenized JSON (for BERT-style input)
```python
import json

with open("preprocessed_reddit_social_issues_cleaned.json", "r") as f:
    data = [json.loads(line) for line in f]

print(data[0])
```

---

## 🔧 Potential Use Cases

- Training **text classification** models to identify infrastructure complaints
- Monitoring **public sentiment** and urgency around social issues
- Supporting **urban policy** and governance research
- Evaluating **NLP models** on real-world, multilingual or noisy text data

---

## 📚 License

Please ensure compliance with Reddit’s [API Terms of Use](https://www.redditinc.com/policies/data-api-terms).

---

## 👤 Author

Prepared by:Haseeb Ahmed  
Contact:engr.haseebahmed103332@gmail.com
