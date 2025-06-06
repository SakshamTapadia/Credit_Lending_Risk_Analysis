from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
import joblib

def encode_target(y):
    """Encode target variable"""
    le = LabelEncoder()
    y = le.fit_transform(y)
    return y

def prepare_data(df, target_col):
    """Split data into train/test sets"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    y = encode_target(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_baseline_model(X_train, y_train, X_test, y_test):
    """Train and evaluate multiple baseline models"""
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost': XGBClassifier(objective='multi:softmax', num_class=4, random_state=42),
        'CatBoost': CatBoostClassifier(iterations=200, random_state=42, verbose=0)
    }
    
    best_score = -1
    best_model = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = (accuracy_score(y_test, y_pred) + f1_score(y_test, y_pred, average='macro')) / 2
        
        if score > best_score:
            best_score = score
            best_model = model
    
    return best_model

def evaluate_model(model, X, y):
    """Evaluate model performance"""
    y_pred = model.predict(X)
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y, y_pred, average='macro'):.4f}")
    print(classification_report(y, y_pred))

def hyperparameter_tuning(best_model, X_train, y_train):
    """Perform hyperparameter tuning"""
    param_distributions = {
        RandomForestClassifier: {
            'n_estimators': randint(100, 500),
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 4)
        },
        XGBClassifier: {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        },
        CatBoostClassifier: {
        'iterations': [300, 350, 400],
        'depth': [5, 6, 10],
        'learning_rate': [0.15, 0.17, 0.2],
        'l2_leaf_reg': [2, 3, 4],
        'border_count': [128, 254],
        'bagging_temperature': [0.5, 1, 2],
        'random_strength': [0.5, 1, 1.5],
        'bootstrap_type': ['Bayesian'],
        'boosting_type': ['Plain'],
        'score_function': ['Cosine'],
        'grow_policy': ['SymmetricTree'],
    }
    }
    
    model_type = type(best_model)
    if model_type not in param_distributions:
        raise ValueError(f"Unsupported model type for tuning: {model_type}")
    
    search = RandomizedSearchCV(
        best_model,
        param_distributions[model_type],
        n_iter=15,
        cv=3,
        scoring='f1_macro',
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    return search.best_estimator_

def save_model(model, path):
    """Save trained model"""
    joblib.dump(model, path)