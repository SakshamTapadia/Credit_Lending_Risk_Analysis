# **Credit Default Prediction ML**

*A Machine Learning model to predict credit default risk using CatBoost*

---

## **📌 Overview**

This project predicts the likelihood of credit default using a **CatBoost Classifier**. It includes:

- **Data preprocessing** (cleaning, merging, encoding)
- **Feature engineering** (handling missing values, categorical features)
- **Model training & evaluation** (CatBoost with hyperparameter tuning)
- **Streamlit web app** for interactive predictions

---

## **📂 Directory Structure**

```plaintext
prime-gamer-credit_default_prediction_ml/
├── config.yml               # Configuration file
├── main.py                  # Main execution script
├── notes.txt                # Project notes
├── requirements.txt         # Python dependencies
├── sample.csv               # Sample dataset
├── catboost_info/           # CatBoost training logs
├── data/                    # Raw & processed datasets
│   ├── processed/           # Cleaned & encoded data
│   └── raw/                 # Original datasets (Excel files)
├── models/                  # Trained model (classifier.pkl)
├── notebooks/               # Jupyter Notebook for EDA
├── src/                     # Source code modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── raw_pipeline_complt.py
│   └── utils.py
└── streamlit_app/           # Streamlit web app
    └── app.py               # Streamlit UI & prediction logic
```

---

## **🖼️ Screenshots**

### **1️⃣ Before Prediction**

![Streamlit](Screenshot_20250523_005912.png)
📌 *Example:* User fills in customer details (income, credit score, etc.).

### **2️⃣ After Prediction**

![Streamlit](Screenshot_20250523_010137.png)
📌 *Example:* Model outputs **P1, P2, P3, P4** categories

---

## **🚀 Quick Start**

### **1. Clone the Repository**

```bash
git clone https://github.com/Prime-Gamer/Credit_Default_Prediction_ML.git
cd Credit_Default_Prediction_ML
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit App**

```bash
streamlit run streamlit_app/app.py
```

*(Open `http://localhost:8501` in your browser)*

### **4. Train the Model (Optional)**

```bash
python main.py
```

*(Uses `config.yml` for settings.)*

---

## **🛠️ Tech Stack**

- **Python** (Pandas, NumPy, Scikit-learn)
- **CatBoost** (Gradient Boosting for classification)
- **Streamlit** (Web app deployment)
- **Jupyter Notebook** (EDA & analysis)

---

## **📝 Notes**

- Data sourced from `case_study1.xlsx` and `case_study2.xlsx`.
- Model performance metrics (accuracy, F1-score) can be found in `notebooks/eda.ipynb`.
- For large datasets, enable GPU in CatBoost (`task_type="GPU"` in `config.yml`).

---

## **📜 License**

MIT License - See [LICENSE](LICENSE) 

---

**👤 Author:** [Saksham Tapadia](https://github.com/SakshamTapadia)
**🔗 Repo:** [GitHub Link](https://github.com/SakshamTapadia/Credit_Lending_Risk_Analysis)

---
