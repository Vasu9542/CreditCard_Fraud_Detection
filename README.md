# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using **Logistic Regression**.

---

## 📌 Project Overview
This project demonstrates:
- Loading and cleaning credit card transaction data
- Handling missing values
- Balancing imbalanced datasets with under-sampling
- Training a Logistic Regression model for fraud detection
- Evaluating model performance using accuracy score and visualizations

---

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn

---

## 📂 Dataset
The dataset used is `creditcard.csv` containing anonymized transaction data labeled as:
- `0` – Legitimate transaction
- `1` – Fraudulent transaction

---

## ⚙️ Workflow

### 1️⃣ Import Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

2️⃣ Load Dataset
Always show details
cc_data = pd.read_csv('creditcard.csv')

3️⃣ Data Cleaning

Checked for missing values

Removed rows with null values

Always show details
creditcard_data = cc_data.dropna()

4️⃣ Exploratory Data Analysis

Distribution of legitimate vs fraudulent transactions

Statistical comparison of transaction amounts

5️⃣ Handle Imbalanced Data (Under-sampling)
Always show details
legit_sample = legit.sample(n=493)
N_Dataset = pd.concat([legit_sample, fraud], axis=0)

6️⃣ Split Data into Training & Testing Sets
Always show details
X = N_Dataset.drop(columns='Class', axis=1)
y = N_Dataset['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

7️⃣ Train Logistic Regression Model
Always show details
model = LogisticRegression()
model.fit(X_train, y_train)

8️⃣ Evaluate Model
Always show details
X_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, X_train_pred)
X_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, X_test_pred)
print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

📊 Visualizations
1️⃣ Class Distribution
Always show details
import seaborn as sns
sns.countplot(x='Class', data=creditcard_data)
plt.title('Class Distribution')
plt.show()

2️⃣ Transaction Amount Distribution
Always show details
legit.Amount.describe()
fraud.Amount.describe()

3️⃣ ROC Curve & Confusion Matrix
Always show details
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

y_pred_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

cm = confusion_matrix(y_test, model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

```
These plots visually show dataset imbalance, model performance, and classification quality.

🚀 How to Run

Clone this repository

Place creditcard.csv in the project folder

Install dependencies:

Always show details
pip install numpy pandas scikit-learn matplotlib seaborn


Run the notebook or Python script

📌 Future Improvements

Try Random Forest or XGBoost

Use SMOTE oversampling instead of under-sampling

Add Precision, Recall, F1-score, ROC-AUC metrics

