# 📊 User Satisfaction Analysis on Deepseek Reviews Using RoBERTa and BERTopic

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![RoBERTa](https://img.shields.io/badge/Model-RoBERTa-orange?logo=huggingface)
![BERTopic](https://img.shields.io/badge/Topic%20Modeling-BERTopic-green)
![NLP](https://img.shields.io/badge/Field-NLP-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Undergraduate Thesis — Information Systems, Universitas Negeri Semarang (2025)  
> Author: **Satria Imawan** | Supervisor: Yahya Nur Ifriza, S.Pd., M.Kom.

---

## 📌 Overview

This research analyzes user satisfaction with the **Deepseek** generative AI chatbot application by integrating two NLP approaches:

- **Sentiment Analysis** using RoBERTa (`w11wo/indonesian-roberta-base-sentimen-classifier`)
- **Topic Modeling** using BERTopic

The dataset consists of **3,614 user reviews** scraped from the Google Play Store (Indonesian regional accounts, January–September 2025). A key novelty of this research is the addition of **app version** as a segmentation variable alongside sentiment, enabling **topic evolution analysis** across application updates.

---

## 🔑 Key Findings

| Metric | Result |
|---|---|
| Total Reviews Analyzed | 3,614 |
| Positive Sentiment | 52.6% |
| Negative Sentiment | 38.9% |
| Neutral Sentiment | 8.5% |
| Topics Identified (raw) | 32 |
| Topics after Consolidation | 11 |
| RoBERTa Accuracy | 97.23% |
| RoBERTa F1-Score | 97.26% |

**Top Topics:**
- 🏆 *Application Quality* — largest topic (1,256 reviews, 95.1% positive)
- ⚠️ *Server Quality* — primary pain point (1,036 reviews, 65.5% negative)

**Overall Verdict:** Deepseek has a strong product-market fit, but server infrastructure issues pose a significant risk to long-term user retention.

---

## 🔬 Research Pipeline

```
Data Collection (Google Play Scraper)
        ↓
Preprocessing
  ├── Case Folding
  ├── Cleansing (HTML, URLs, special chars, emoji → text)
  └── Normalization (slang/informal → standard Bahasa Indonesia)
        ↓
Exploratory Data Analysis
  ├── Word Cloud
  └── Frequency Analysis
        ↓
Sentiment Classification (RoBERTa)
  └── Labels: Positive / Neutral / Negative
        ↓
Segmentation (Sentiment × App Version)
        ↓
Topic Modeling (BERTopic)
  ├── Embedding: paraphrase-multilingual-MiniLM-L12-v2 (SBERT)
  ├── Dimensionality Reduction: UMAP
  ├── Clustering: HDBSCAN
  └── Topic Representation: c-TF-IDF
        ↓
Topic Evolution Analysis (per app version)
        ↓
Interpretation & Conclusions
```

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.10+ |
| Data Collection | `google-play-scraper` |
| Data Processing | `pandas`, `numpy`, `regex` |
| NLP / Sentiment | `transformers`, `torch`, `HuggingFace` |
| Sentiment Model | `w11wo/indonesian-roberta-base-sentimen-classifier` |
| Topic Modeling | `bertopic`, `sentence-transformers` |
| Dimensionality Reduction | `umap-learn` |
| Clustering | `hdbscan` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud`, `plotly` |

---

## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Raw scraped reviews (CSV)
│   └── processed/              # Preprocessed data
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_sentiment_analysis.ipynb
│   └── 05_topic_modeling.ipynb
├── outputs/
│   ├── figures/                # Visualizations
│   └── results/                # Topic & sentiment results (CSV)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/deepseek-review-analysis.git
cd deepseek-review-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebooks
Execute notebooks sequentially from `01` to `05` inside the `notebooks/` folder.

---

## 📊 Model Evaluation

### RoBERTa Sentiment Classifier
| Metric | Score |
|---|---|
| Accuracy | 97.23% |
| Precision | 97.33% |
| Recall | 97.23% |
| F1-Score | 97.26% |

### BERTopic
| Metric | Score |
|---|---|
| Topic Coherence | High (close to 1.0) |
| Topic Diversity | High (close to 1.0) |

---

## 🌟 Research Novelty

Most prior studies treat app reviews as a static corpus. This research introduces **dual segmentation** — combining **sentiment polarity** and **app version** — to enable dynamic topic evolution analysis. This allows detection of:

- New issues introduced in specific app updates
- Verification of whether bug fixes were effective
- Feature adoption patterns across versions

---

## 📚 Key References

- Liu et al. (2019) — RoBERTa: A Robustly Optimized BERT Pretraining Approach
- Grootendorst (2022) — BERTopic: Neural topic modeling with a class-based TF-IDF
- Vaswani et al. (2017) — Attention Is All You Need
- DeLone & McLean (2003) — Information System Success Model

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Satria Imawan**  
Information Systems — Universitas Negeri Semarang  
📧 [LinkedIn](https://linkedin.com/in/your-profile) | 🐙 [GitHub](https://github.com/your-username)
