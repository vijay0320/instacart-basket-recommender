# 🛒 Instacart Basket-Level Recommendation System

A **production-style basket recommendation system** built on the Instacart Market Basket Analysis dataset. This project goes beyond basic EDA and association rules to build a **learning-based recommender** that suggests complementary products based on the *entire basket context*.

Designed to demonstrate skills relevant to **Data Analyst, Business Analyst, Data Engineer, and Entry-Level ML roles**.

---

## 📌 Problem Statement

> *Given a user’s current shopping basket, which additional products should be recommended to maximize relevance and cross-sell potential?*

Traditional item-to-item recommendations often fail to consider **basket context**. This project models basket-level semantics using learned product embeddings and contextual similarity.

---

## 🧠 Solution Overview

The solution progresses through **four analytical layers**:

1. **Exploratory Data Analysis (EDA)**
   Understand user ordering behavior, reorder patterns, time-based trends, and department-level insights.

2. **Business Insight Generation**
   Translate EDA into actionable insights (habitual products, anchor items, reorder-driven categories).

3. **Baseline Predictive Modeling**
   Logistic regression to predict reorder probability using user–product and temporal features.

4. **Deep Learning Basket Recommender (Core Contribution)**
   A PyTorch-based neural model that learns **product embeddings** and recommends items based on the *entire basket*, not just one product.

---

## 🔍 Key Insights and Analysis

* **~59% of all items are reorders**, highlighting strong habitual behavior
* Dairy, Produce, and Beverages show the **highest reorder affinity**
* Basket size is right-skewed: median ≈ 8 items, mean ≈ 10 items
* Orders placed in the **morning hours** have higher reorder likelihood
* Presence of an **anchor product** significantly increases basket size

---

## 🧪 Modeling Approaches

### 1️⃣ Baseline Model – Logistic Regression

**Objective:** Predict whether a product will be reordered

**Features include:**

* User–product reorder rate
* Days since prior order
* Order day of week & hour
* Anchor-product affinity

**Performance:**

* ROC-AUC ≈ **0.85**
* Strong interpretability for business insights

---

### 2️⃣ Deep Learning Model – Basket Recommender (Main Model)

A neural network that learns **dense embeddings for products** and scores candidate items against the *aggregated basket embedding*.

**Architecture:**

* Product Embedding Layer (shared)
* Masked mean pooling over basket items
* Dot-product similarity with candidate product
* Binary cross-entropy loss with negative sampling

```text
Basket → [Product Embeddings] → Basket Vector
Candidate Product → Embedding
Score = Similarity(Basket Vector, Product Vector)
```

---

## 🤖 Interactive Basket Recommender

An interactive CLI demo allows users to:

* Add products by name
* Build a basket incrementally
* Receive **2 context-aware recommendations** for the entire basket

Example:

```
Added: Organic Milk
Recommended pairings:
1. Grade A Large Eggs
2. Parmesan Cheese
```

This mirrors real-world grocery UX where only **1–2 high-confidence recommendations** are shown.

---

## 🛠️ Tech Stack

* **Python**
* **Pandas / NumPy** – data processing
* **Matplotlib / Seaborn** – visualization
* **Scikit-learn** – baseline modeling
* **PyTorch** – deep learning recommender
* **Jupyter Notebook** – analysis & experimentation
* **Git / GitHub** – version control

---

## 📂 Project Structure

```
instacart-basket-recommender/
│
├── notebook.ipynb        # Full analysis + modeling pipeline
├── models/               # Saved model & embeddings
├── .gitignore            # Excludes large datasets
├── README.md             # Project documentation
```

> ⚠️ Raw datasets (~4GB) are excluded from GitHub. Instructions to obtain them are included below.

---

## 📥 Dataset

**Source:** Instacart Market Basket Analysis (Kaggle)

To reproduce results:

1. Download the dataset from Kaggle @ https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis
2. Place CSV files inside a local `datasets/` folder
3. Run the notebook top-to-bottom

---

## 💾 Model Saving

The trained model and embeddings are saved using PyTorch:

```python
torch.save(model.state_dict(), "models/basket_recommender.pt")
```

These embeddings can be reused for:

* Nearest-neighbor recommendations
* Clustering similar products
* Downstream personalization tasks

---

## 🎯 Skills Demonstrated

* End-to-end analytics project ownership
* Business problem framing
* Large-scale data handling (30M+ rows)
* Feature engineering & EDA
* Supervised ML & evaluation
* Deep learning for recommendations
* Practical system design decisions

---

## 🚀 Future Improvements

* Candidate generation optimization (ANN / FAISS)
* Session-based or sequential modeling
* User embeddings for personalization
* Offline ranking metrics (Recall@K, NDCG)
* Lightweight API for real-time inference

---

## 👤 Author

**Vijay Aditya**
Aspiring Data / Business / ML Analyst

---

⭐ If you found this project useful or insightful, feel free to star the repo!
