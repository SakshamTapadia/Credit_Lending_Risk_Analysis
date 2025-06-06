import pandas as pd
import numpy as np

def load_data(file1_path, file2_path):
    """Load and merge the two Excel files"""
    df1 = pd.read_excel(file1_path)
    df2 = pd.read_excel(file2_path)
    return df1, df2

def clean_data(df1, df2):
    """Clean the raw data"""
    # Clean df1
    df1 = df1[df1['Age_Oldest_TL'] >= 0]
    
    # Clean df2
    columns_to_drop = [col for col in df2.columns if df2[df2[col] == -99999].shape[0] > 10000]
    df2 = df2.drop(columns=columns_to_drop)
    
    for col in df2.columns:   
        df2 = df2[df2[col] != -99999]
    
    return df1, df2

def merge_data(df1, df2):
    """Merge the two dataframes"""
    common_cols = df1.columns.intersection(df2.columns).tolist()
    df = pd.merge(df1, df2, left_on=common_cols, how='inner', right_on=common_cols)
    return df

def save_data(df, path):
    """Save dataframe to CSV"""
    df.to_csv(path, index=False)