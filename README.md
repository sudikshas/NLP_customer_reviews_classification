# NLP_customer_reviews_classification

<div align="center">

# 🧠 NLP Sentiment Classification of Company Reviews
### Benchmarking Classical ML vs. Deep Learning Architectures for Customer Feedback Analysis

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![UC Berkeley](https://img.shields.io/badge/UC%20Berkeley-W207%20Applied%20ML-003262?style=flat-square)](https://ischool.berkeley.edu)

</div>

---

## 📌 Project Overview

This project presents a **rigorous end-to-end NLP pipeline** that benchmarks five machine learning architectures — from Logistic Regression to Bidirectional LSTM — on binary sentiment classification of 100,000 real-world customer reviews spanning 45 companies. The goal was not just to build a classifier, but to answer a precise research question:

> **How does model complexity — from bag-of-words statistical models to deep sequence architectures — affect generalization accuracy on customer sentiment data?**

The project covers the full data science lifecycle: exploratory data analysis, class balancing, text preprocessing, model development, hyperparameter ablation studies, external domain validation, and learned embedding visualization.

**Best result: 96.3% test accuracy** with a Bidirectional LSTM — a **30-point improvement** over the Logistic Regression baseline.

---

## 🚨 Problem Statement

Businesses generate millions of customer reviews that cannot be manually analyzed at scale. Automated sentiment classification enables:

- Real-time brand health monitoring across product lines
- Feedback-driven recommendation engines (Amazon attributes 35% of revenue to its feedback-based engine)
- Early detection of product defects or service failures before they compound
- Quantified customer satisfaction metrics for executive dashboards

> *McKinsey research shows companies that leverage customer analytics extensively are **2x as likely to generate above-average profits and sales growth.***

The technical challenge: which model architecture best captures the semantic complexity of review language while generalizing to unseen data — and **why**?

---

## 📂 Dataset

| Attribute | Detail |
|---|---|
| **Primary source** | Kaggle NLP Company Reviews Dataset |
| **Raw size** | 100,000 reviews from **45 companies** |
| **Features** | `Id`, `Review` (raw text), `Rating` (1–5 stars) |
| **Training set** | 24,000 balanced reviews (12K positive, 12K negative) |
| **Split** | 60% train / 20% validation / 20% test |
| **External validation** | Glassdoor reviews — Integral Ad Science, Lockheed Martin, Veeva Systems, Splunk |

### Label Engineering

Star ratings were mapped to sentiment classes with a deliberate design decision:

```
Ratings 1–2  →  "negative"  (0)
Rating  3    →  "neutral"   → DROPPED (too ambiguous for binary signal)
Ratings 4–5  →  "positive"  (1)
```

Neutral reviews were excluded to eliminate training noise from ambiguous labels — a clean binary signal was prioritized over volume.

### Class Balancing

The raw dataset was imbalanced. Downsampling to **12,000 examples per class** eliminated bias without oversampling artifacts:

```python
temp_positive = df_balanced[df_balanced.Sentiment == 'positive'].sample(n=12000, replace=False)
temp_negative = df_balanced[df_balanced.Sentiment == 'negative'].sample(n=12000, replace=False)
df_balanced = pd.concat([temp_positive, temp_negative]).sample(frac=1)
```

### Key EDA Findings

| Metric | Positive Reviews | Negative Reviews |
|---|---|---|
| Average length | **28 words** | **110 words** |
| Dominant length range | 0–50 words | 0–150 words |
| Long-text dominance | Rare | Consistently high |

> **Insight:** Negative reviews are ~4× longer. Customers elaborate more when dissatisfied — this directly informed our choice of `max_sequence_length=400` for deep learning models to avoid truncating negative review context.

---

## ⚙️ Approach

### Text Preprocessing Pipeline

All reviews went through a consistent cleaning pipeline before feature extraction:

```python
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)                          # Strip HTML tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)  # Preserve emoticons
    text = re.sub('[\W]+', ' ', text.lower())                  # Lowercase + remove non-alpha
    text += ' '.join(emoticons).replace('-', '')               # Re-append emoticons
    return text
```

Labels were binary-encoded: `positive → 1`, `negative → 0`.

**For classical ML:** `TfidfVectorizer(max_features=1000)` produced numerical feature matrices.

**For deep learning:** `Tokenizer` + `pad_sequences` converted text to integer-indexed sequences of fixed length (50 for LSTM, 400 for BiLSTM and CNN).

---

### Model Architecture Summary

#### 🔹 Model 1 — Logistic Regression (Baseline)
A TF-IDF bag-of-words representation fed into sklearn's Logistic Regression. Establishes interpretable performance floor.
```
TF-IDF (1,000 features) → LogisticRegression(max_iter=100)
```

#### 🔹 Model 2 — Random Forest
Same TF-IDF feature space, ensemble tree approach to test whether variance reduction closes the generalization gap.
```
TF-IDF (1,000 features) → RandomForestClassifier(n_estimators=40, max_depth=20)
```

#### 🔹 Model 3 — LSTM
First deep learning model. Learned embeddings replace hand-crafted TF-IDF features; sequential processing replaces bag-of-words.
```
Embedding(10K vocab, 100-dim, max_len=50) → LSTM(64, dropout=0.2) → Dense(sigmoid)
```
Trained with Adam, binary cross-entropy, EarlyStopping on `val_loss`.

#### 🔹 Model 4 — Bidirectional LSTM ⭐ Best Model
Processes the input sequence both forward and backward, capturing left and right context simultaneously.
```
Embedding(1K vocab, 10-dim, max_len=400)
→ BiLSTM(64 units, dropout)
→ BiLSTM(32 units, dropout)
→ Dense(64, ReLU)
→ Dense(1, sigmoid)
```
Adam (lr=0.001), binary cross-entropy, 15 epochs.

#### 🔹 Model 5 — 1D CNN
Convolutional filters learn local n-gram patterns (phrase-level features) — no sequential processing required.
```
Embedding(1K vocab, 10-dim, max_len=400)
→ Conv1D(126 filters, k=4, ReLU) → Dropout → MaxPooling1D(2)
→ Conv1D(64 filters, k=4, ReLU) → Dropout → MaxPooling1D(2)
→ GlobalAveragePooling1D
→ Dense(1, sigmoid)
```
Adam (lr=0.001), binary cross-entropy, 10 epochs.

---

### Hyperparameter Ablation Studies

Three structured experiments were run to understand the sensitivity of model performance to architectural choices:

| Experiment | Change | Result | Conclusion |
|---|---|---|---|
| BiLSTM — Dropout removal | Removed dropout from both BiLSTM layers | Val loss increased after epoch 10; training loss dropped | Dropout is essential for recurrent regularization |
| BiLSTM — Scale up | vocab 1K→10K, embedding 10→150-dim, max_len 400→200 | Train acc 99.9%, val loss showed increasing trend | Larger vocab without more data causes memorization |
| CNN — Depth | Added 3rd Conv1D layer | No meaningful change in accuracy or loss | 2-layer CNN already captures relevant n-gram patterns |
| All DL — Embedding dim | Tested 8-dim → 10-dim → 100-dim | Negligible accuracy difference across all | Task is not representation-bottlenecked at these scales |

---

### Embedding Visualization

Learned word embeddings from the CNN were extracted and projected to 2D using PCA, then visualized interactively with Plotly — confirming that semantically related words cluster in the embedding space.

```python
embeddings_cnn = model_cnn.get_layer('embedding').get_weights()[0]
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_cnn)
```

---

## 📊 Results

### Performance Benchmark

| Model | Train Acc | Train Loss | Val Acc | Test Acc | Test Loss | ∆ vs. Baseline |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.967 | 0.111 | 0.679 | 0.663 | 0.635 | — |
| Random Forest | 0.965 | 0.192 | 0.762 | 0.801 | 0.468 | +13.8pp |
| LSTM | 0.985 | 0.046 | 0.954 | 0.952 | 0.161 | +28.9pp |
| **Bidirectional LSTM** | **0.977** | **0.069** | **0.963** | **0.963** | **0.111** | **+30.0pp** |
| CNN | 0.968 | 0.097 | 0.959 | 0.961 | 0.128 | +29.8pp |

### Reading the Results

- **Logistic Regression** shows a 30-point train-to-test drop — a textbook overfitting case. TF-IDF features memorize word frequencies without capturing context or word order.
- **Random Forest** closes 14 points of that gap through ensemble averaging, but still fundamentally constrained by the bag-of-words feature space.
- **LSTM** jumps nearly 29 points in test accuracy by learning sequential word relationships — the single biggest architectural improvement.
- **Bidirectional LSTM** achieves the highest test accuracy (96.3%) and the lowest test loss (0.111), with the tightest train-to-test consistency of all models — the best generalizer.
- **CNN** is within 0.2pp of the BiLSTM at 96.1%, demonstrating that local phrase patterns (n-grams via Conv1D filters) capture most of the sentiment signal in short-to-medium review text.

### External Validation — Glassdoor Reviews

All five models were validated on a separately scraped Glassdoor employee review dataset (4 companies), testing generalization to a domain shift: trained on general customer reviews → evaluated on employee sentiment reviews. Deep learning models maintained strong performance; classical models showed larger accuracy drops.

---

## 💡 Key Insights & Learnings

**1. Word order and context carry most of the predictive signal in text.**
The 30-point accuracy jump from bag-of-words (LR: 66.3%) to sequence models (BiLSTM: 96.3%) is the clearest result in this project. Statistical frequency of individual words is a poor proxy for sentiment.

**2. Bidirectional context matters — but not as much as going deep at all.**
BiLSTM outperforms LSTM by 1.1pp. The bigger gain is the jump from classical ML to any sequential model. Going bidirectional provides a meaningful but incremental improvement on top of the already-strong LSTM.

**3. CNNs are highly competitive for short-to-medium text classification.**
The CNN matched BiLSTM to within 0.2pp despite processing no sequential order. Conv1D filters with kernel_size=4 act as learned n-gram detectors — phrases like "terrible experience" or "excellent service" are strong enough local signals to achieve near-BiLSTM performance.

**4. Dropout is non-negotiable in recurrent networks.**
Removing dropout from the BiLSTM caused validation loss to diverge after epoch 10 while training loss continued falling — a textbook example of recurrent network overfitting that validates dropout's role as a regularizer for sequential models.

**5. Hyperparameter tuning has diminishing returns once accuracy is high.**
Scaling up vocab size, embedding dimensions, and sequence length produced marginal accuracy changes — and in some cases introduced overfitting. When a model already generalizes at 96%, the bottleneck is elsewhere (data diversity, pre-trained representations, task framing).

**6. Negative reviews are linguistically more complex.**
At 4× the average length of positive reviews (110 vs. 28 words), negative feedback requires models capable of long-range dependency capture. This is a generalizable business insight: customers who are unhappy elaborate more, making their feedback richer in signal — and harder to model.

**7. Domain transfer is achievable without retraining.**
Models trained on general company reviews generalized meaningfully to Glassdoor employee reviews — suggesting the learned sentiment representations capture generalizable linguistic patterns rather than domain-specific vocabulary.

---

## 📈 Business Impact

| Business Use Case | How This Model Enables It |
|---|---|
| **Brand health monitoring** | Classify millions of incoming reviews in real time — no human labeling required |
| **Product feedback loop** | Identify product-specific negative sentiment spikes before they compound |
| **Customer support triage** | Prioritize high-urgency negative reviews automatically for faster resolution |
| **Competitive benchmarking** | Compare sentiment distributions across companies using the same model |
| **Recommendation engine input** | Positive sentiment signals feed personalization algorithms directly |

A system at 96.3% accuracy operating on 1 million reviews per month would correctly classify **963,000 reviews** — with only ~37,000 misclassifications. At human review costs of even $0.10/review, automating this classification represents **$100,000/month in annotation cost savings** per million reviews.

---

## ⚠️ Limitations & Future Work

### Current Limitations

- **Binary framing discards neutral signal.** Three-star reviews were dropped to ensure clean labels, but mixed sentiment contains actionable feedback that the current model cannot access.
- **No pre-trained embeddings.** All embedding layers were trained from scratch on 24,000 samples with a 1,000-token vocabulary — a significant representation bottleneck. GloVe, fastText, or BERT would substantially close remaining error.
- **Minimal text normalization.** No stemming, lemmatization, or stopword removal was applied. Standard preprocessing would improve vocabulary coverage and reduce out-of-vocabulary rates.
- **No aspect-level sentiment.** The model classifies overall review tone, not *what specifically* the customer liked or disliked (shipping, product quality, customer service).
- **LSTM truncation at max_len=50.** The initial LSTM used a 50-token window, but the average negative review is 110 words — meaning substantial context was systematically truncated for the worst-case sentiment class.

### Planned Extensions

- [ ] Fine-tune `bert-base-uncased` on this task to establish a modern performance ceiling
- [ ] Add aspect-based sentiment analysis (ABSA) to extract entity-level sentiment
- [ ] Build multi-class classification preserving the neutral/mixed sentiment category
- [ ] Deploy the BiLSTM as a REST API (FastAPI + Docker) for real-time inference
- [ ] Implement SHAP or LIME for model interpretability — which tokens drive each prediction?
- [ ] Extend to multilingual reviews using multilingual BERT

---

## 🛠️ Skills & Tech Stack

### Machine Learning & NLP
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-4A90D9?style=flat-square)

| Skill | Applied In |
|---|---|
| **Recurrent Neural Networks (LSTM, BiLSTM)** | Sequence modeling with long-range dependency capture |
| **Convolutional Neural Networks (1D CNN)** | Local n-gram feature extraction for text classification |
| **TF-IDF Vectorization** | Bag-of-words feature engineering for classical ML |
| **Tokenization & Sequence Padding** | Deep learning text preprocessing pipeline |
| **Word Embedding Learning** | End-to-end embedding layer training |
| **Ensemble Methods (Random Forest)** | Variance reduction over single-tree classifiers |
| **Hyperparameter Ablation** | Systematic architecture sensitivity analysis |
| **Class Balancing (Downsampling)** | Correcting label imbalance without oversampling artifacts |
| **PCA Embedding Visualization** | Dimensionality reduction for learned representation analysis |
| **Binary Cross-Entropy Optimization** | Loss function choice for binary classification |
| **EarlyStopping / Regularization** | Dropout, patience-based stopping for generalization |
| **Domain Generalization Testing** | External validation on Glassdoor dataset |

### Data & Visualization
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Viz-4C8CBF?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat-square&logo=plotly&logoColor=white)

### Environment
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Training-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

---

## 📁 Repository Structure

```
├── Sentiment_Classification_Models_Part_1.ipynb   # Baseline models: LR, RF, LSTM + Glassdoor validation
├── Sentiment_Classification_Models_Part_2.ipynb   # Deep learning: BiLSTM, CNN, ablations, embedding viz
├── W207_Final_Project_Slides.pdf                  # Project presentation deck
├── data/
│   ├── train.csv                                  # Primary training data (100K reviews)
│   └── Final_Company_Reviews.csv                  # Glassdoor external validation set
└── README.md
```

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/sudikshas/NLP_customer_reviews_classification.git
cd nlp-sentiment-classification

# Install dependencies
pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn plotly nltk wordcloud

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

# Run notebooks in order
jupyter notebook Sentiment_Classification_Models_Part_1.ipynb
jupyter notebook Sentiment_Classification_Models_Part_2.ipynb
```

---

Built as a collaborative final project for **W207: Applied Machine Learning** at UC Berkeley School of Information (MIDS Program).

---

<div align="center">

*W207 Applied Machine Learning · UC Berkeley School of Information*

</div>
