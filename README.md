# 🌾 Classifying Farm Classified Ads using Naive Bayes & SVM

This predictive modeling project classifies online classified advertisements as **relevant or non-relevant** to the farming community. The dataset contains real ad listings from a farming portal, with goals to reduce spam, improve ad quality, and enhance user experience.

---

## 🗃️ Dataset

**File:** `FarmAds.csv`  
Each row includes:
- `doc_id`: Unique ad identifier
- `text`: Ad content (unstructured text)
- `label`: Classification (-1 = not relevant, 1 = relevant)

---

## ⚙️ Workflow Overview

### 1. 🧹 Data Preprocessing
- Converted ad text into a **corpus** using `tm` and `SnowballC`
- Applied:
  - Lowercasing
  - Punctuation removal
  - Stop word filtering
  - Whitespace stripping
  - Stemming
- Built a **Document-Term Matrix (DTM)** and reduced sparsity
- Applied **TF-IDF weighting** for term importance

---

### 2. 📊 Data Partitioning
- Split into **80% training**, **20% testing**
- Reproducible split using `set.seed(1947)`
- Used `createDataPartition()` from the `caret` package

---

### 3. 🤖 Model Training & Evaluation

#### 🟦 Naive Bayes (klaR)
- Accuracy: **72.87%**
- Sensitivity (True Positives): **93.61%**
- Specificity (True Negatives): **54.75%**
- Kappa: **0.47** — moderate agreement

#### 🟧 Support Vector Machine (SVM)

- **Linear Kernel (vanilladot)**
  - Accuracy: **89.05%**
  - Sensitivity: **93.78%**
  - Specificity: **84.92%**
  - Kappa: **0.78**

- **RBF Kernel (rbfdot)**
  - Accuracy: **88.33%**
  - Sensitivity: **87.05%**
  - Specificity: **89.44%**
  - Kappa: **0.76**

---

## ✅ Conclusion

- **SVM with Linear Kernel** achieved the best balance of accuracy and interpretability.
- **RBF Kernel** performed nearly as well but offered stronger specificity.
- Preprocessing text effectively is crucial for performance in text classification tasks.

---

## 📦 R Libraries Used
```r
library(tm)
library(SnowballC)
library(caret)
library(klaR)
library(kernlab)
