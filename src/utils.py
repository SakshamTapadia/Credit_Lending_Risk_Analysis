import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st



def plot_correlation_matrix(df):
    """Plot correlation matrix"""
    plt.figure(figsize=(15, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def get_feature_importance(model, feature_names):
    """Get and display feature importance"""
    importances = model.get_feature_importance()
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(20))
    plt.title('Top 20 Important Features')
    plt.show()
    
    return feat_imp_df