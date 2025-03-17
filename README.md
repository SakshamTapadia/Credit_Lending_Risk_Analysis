# 📌 Credit Lending Risk Analysis

## 📖 Project Overview
This project aims to analyze credit risk and predict loan approval decisions using **machine learning models**. It involves understanding financial risk concepts such as **Non-Performing Assets (NPA), Days Past Due (DPP), and Portfolio at Risk (PAR)** to improve loan approval processes.

## 📂 Directory Structure
Directory structure:
└── sakshamtapadia-credit_lending_risk_analysis/
    ├── bank_notes.docx
    ├── case_study1.xlsx
    ├── case_study2.xlsx
    ├── Features_Target_Description.xlsx
    ├── Main.ipynb
    └── project_stats.ipynb

## 📊 Dataset Description
The dataset includes **customer financial details** such as **credit history, income, loan amount, repayment behavior**, and **days past due (DPP)**.

### 🔹 Features:
- **Numerical Features:** Loan amount, income, outstanding principal, DPP, time since last payment, etc.
- **Categorical Features:** Loan type, marital status, gender, education level, last product inquiry, etc.
- **Target Variable:** `Approved_Flag` (P1, P2, P3, P4) - Represents different risk categories.

## 🔍 Problem Statement
Predict whether a loan should be approved or rejected based on the **risk profile** of the borrower using **Decision Trees, Random Forest, XGBoost, and Naïve Bayes**.

---

## 🚀 Implementation Steps
### **1️⃣ Exploratory Data Analysis (EDA)**
- Identified missing values, outliers, and distribution of features.
- Analyzed **credit risk indicators** such as **NPA, DPP, OSP, and PAR**.

### **2️⃣ Feature Engineering**
- Converted categorical variables using **One-Hot Encoding & Ordinal Encoding**.
- Scaled numerical features using **StandardScaler & MinMaxScaler**.

### **3️⃣ Model Training**
Trained multiple ML models to classify loans into different risk categories:
- **Random Forest**
- **XGBoost**
- **Decision Tree**
- **Naïve Bayes**

### **4️⃣ Model Evaluation**
- Used **accuracy, precision, recall, and F1-score**.
- Plotted **Confusion Matrices** for performance visualization.
- Conducted **Cross-Validation & Hyperparameter Tuning** for XGBoost.

---

## ⚙️ How to Run the Project
### **📌 Prerequisites**
1. Install dependencies:
   ```sh
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
2. Open Main.ipynb in Jupyter Notebook or run
   jupyter notebook Main.ipynb

📌 Running the Machine Learning Pipeline
Execute the following steps in Main.ipynb:

Load the dataset.
Preprocess features using scaling & encoding.
Train multiple models & compare performance.
Fine-tune XGBoost with GridSearchCV.
Evaluate models with Confusion Matrix & Classification Report.
📈 Results & Insights
XGBoost achieved the highest accuracy in risk prediction.
Random Forest performed well but required hyperparameter tuning.
Decision Tree was interpretable but prone to overfitting.
Naïve Bayes struggled due to non-Gaussian distributions in financial data.
🔹 Future Improvements
✔️ Use Deep Learning (Neural Networks) for better classification.
✔️ Integrate feature selection techniques for improved accuracy.
✔️ Develop an API for real-time risk prediction in banking applications.

📚 References
bank_notes.docx (Contains banking concepts & risk factors)
case_study1.xlsx & case_study2.xlsx (Financial datasets for analysis)
Features_Target_Description.xlsx (Detailed feature descriptions)

💡 Author
Saksham Tapadia
📧 Email: [sakshamtapadia10@gmail.com]
🔗 GitHub: [https://github.com/SakshamTapadia]
🚀 LinkedIn: [https://www.linkedin.com/in/saksham-tapadia/]


📌 Note: This project is for educational purposes and aims to improve credit risk assessment using machine learning techniques.
