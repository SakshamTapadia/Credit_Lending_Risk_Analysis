# src/main.py
import joblib
import yaml
import pandas as pd
from src.data_preprocessing import load_data, clean_data, merge_data, save_data
from src.feature_engineering import select_features, encode_features
from src.model_training import prepare_data, train_baseline_model, evaluate_model, hyperparameter_tuning, save_model
from src.utils import plot_correlation_matrix, get_feature_importance

def load_config(config_path='config.yml'):
    """Load configuration file"""
    with open(config_path) as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config()
    
    # 1. Data Preprocessing
    print("Loading and preprocessing data...")
    df1, df2 = load_data(config['data']['raw_data1'], config['data']['raw_data2'])
    df1_clean, df2_clean = clean_data(df1, df2)
    merged_df = merge_data(df1_clean, df2_clean)
    save_data(merged_df, config['data']['processed_data'])
    
    # 2. Feature Engineering
    print("\nPerforming feature engineering...")
    target_col = 'Approved_Flag'
    useful_num_cols, useful_cat_cols = select_features(merged_df, target_col)
    features = useful_num_cols + useful_cat_cols + [target_col]
    selected_df = merged_df[features].copy()
    
    encoded_df = encode_features(selected_df, useful_cat_cols)
    save_data(encoded_df, config['data']['encoded_data'])
    
    # 3. Model Training
    print("\nPreparing data for modeling...")
    encoded_df = pd.read_csv(config['data']['encoded_data'])
    merged_df = pd.read_csv(config['data']['processed_data'])
    target_col = 'Approved_Flag'
    X_train, X_test, y_train, y_test = prepare_data(encoded_df, target_col)
    
    print("\nTraining baseline model...")
    baseline_model = train_baseline_model(X_train, y_train, X_test, y_test)
    
    print("\nBaseline Model Performance:")
    print("\nTrain Set:")
    evaluate_model(baseline_model, X_train, y_train)
    print("\nTest Set:")
    evaluate_model(baseline_model, X_test, y_test)
    
    # 4. Hyperparameter Tuning
    print("\nPerforming hyperparameter tuning...")
    best_model = hyperparameter_tuning(baseline_model, X_train, y_train)
    
    print("\nTuned Model Performance:")
    print("\nTrain Set:")
    evaluate_model(best_model, X_train, y_train)
    print("\nTest Set:")
    evaluate_model(best_model, X_test, y_test)
    
    # 5. Save the best model
    print("\nSaving the best model...")
    save_model(best_model, config['model']['save_path'])
    
    # 6. Visualization and Analysi
    print("\nGenerating visualizations...")
    best_model = joblib.load(config['model']['save_path'])
    encoded_df = pd.read_csv(config['data']['encoded_data'])
    merged_df = pd.read_csv(config['data']['processed_data'])
    target_col = 'Approved_Flag'
    X_train, X_test, y_train, y_test = prepare_data(encoded_df, target_col)
    plot_correlation_matrix(encoded_df)
    feat_imp_df = get_feature_importance(best_model, X_train.columns)
    
    print("\nTop 10 Important Features:")
    print(feat_imp_df.head(10))
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()