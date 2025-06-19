# 🗂️ Reddit Social Issues Dataset

This dataset captures Reddit user discussions related to social infrastructure issues in Asian Cities, focusing on topics like electricity, water, waste management, traffic, and security. It includes both a **cleaned CSV** version and a **tokenized JSON** format ready for machine learning.

---

## 📁 Files Included
| File | Description |
|------|-------------|
| `reddit_social_issues_cleaned.csv` | Cleaned, labeled dataset with structured Reddit posts and metadata |
| `preprocessed_reddit_social_issues_cleaned.json` | Tokenized version of the dataset, ready for transformer-based NLP models (e.g., BERT) |

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

Prepared by:    Owais Ali Khan                        and                                    Haseeb Ahmed  
Contact:        awais.ali.khan610@gmail.com                                                  engr.haseebahmed103332@gmail.com
