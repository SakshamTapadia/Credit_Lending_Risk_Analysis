import pandas as pd
import numpy as np
import scipy.stats as st
from statsmodels.stats.outliers_influence import variance_inflation_factor

def select_features(df, target_col):
    """Select relevant features using statistical tests"""
    # Separate categorical and numerical columns
    cat_cols = df.select_dtypes(include=['object']).columns.drop(target_col).tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('PROSPECTID').tolist()
    
    # Chi-square test for categorical features
    useful_cat_cols = []
    for col in cat_cols:
        p_value = st.chi2_contingency(pd.crosstab(df[col], df[target_col]))[1]
        if p_value <= 0.05:
            useful_cat_cols.append(col)
    
    # VIF for numerical features
    remaining_num_cols = num_cols.copy()
    dropped = True
    while dropped:
        dropped = False
        vif_data = [variance_inflation_factor(df[remaining_num_cols].values, i) 
                   for i in range(len(remaining_num_cols))]
        
        max_vif = max(vif_data)
        if max_vif > 6:
            index = vif_data.index(max_vif)
            remaining_num_cols.pop(index)
            dropped = True
    
    # ANOVA for numerical features
    useful_num_cols = []
    for col in remaining_num_cols:
        groups = [df[col][df[target_col] == category].dropna() 
                 for category in df[target_col].unique()]
        p_value = st.f_oneway(*groups)[1]
        if p_value <= 0.05:
            useful_num_cols.append(col)
    
    return useful_num_cols, useful_cat_cols

def encode_features(df, cat_cols):
    """Encode categorical features"""
    # Ordinal encoding for EDUCATION
    education_map = {
        '12TH': 2, 'GRADUATE': 3, 'SSC': 1, 
        'POST-GRADUATE': 4, 'UNDER GRADUATE': 3, 
        'OTHERS': 1, 'PROFESSIONAL': 4
    }
    df['EDUCATION'] = df['EDUCATION'].map(education_map)
    
    # One-hot encoding for other categorical features
    df_encoded = pd.get_dummies(df, columns=[col for col in cat_cols if col != 'EDUCATION'], dtype=int)
    
    return df_encoded