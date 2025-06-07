import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import scipy.stats as st
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import xgboost as xgb
import scikit_posthocs as sp


# 1. EDA



a1 = pd.read_excel('data/raw/case_study1.xlsx')
a2 = pd.read_excel('data/raw/case_study2.xlsx')

df1 = a1.copy()
df2 = a2.copy()

df1.head()
df1.describe()
df1['Age_Oldest_TL'].value_counts().sort_index()
df1['Age_Newest_TL'].value_counts().sort_index()

df1[df1['Age_Oldest_TL'] == 0]

# both Age_Oldest_TL and Age_Newest_TL have negative values as -99999 indicating that it is null and filled only for prevent null value error
df1 = df1[df1['Age_Oldest_TL'] >= 0]

# as there are so many null values in df2, we will drop the columns with more than 10000 null values as filling them is risky
df2.isna().sum().sort_values()

# values are like -9999 indicating that it is null and filled only for prevent null value error

columns_to_drop = []
for i in df2.columns:
    if df2[df2[i] == -99999].shape[0] > 10000:
        columns_to_drop.append(i)

{i: df2[df2[i] == -99999].shape[0] for i in columns_to_drop}

# all columns have more than 10000 null values, so drop them

df2 = df2.drop(columns=columns_to_drop)

# checking for less than 10000 null values
rows_to_remove = dict()
for i in df2.columns:
    if df2[df2[i] == -99999].shape[0] > 0:
        rows_to_remove[i] = df2[df2[i] == -99999].shape[0]

# maximum null values or say -99999 are 6321, so these rows can be dropped
for i in df2.columns:   
    df2 = df2[df2[i] != -99999]

# again checking for null values

rows_to_remove2 = dict()
for i in df2.columns:
    if df2[df2[i] == -99999].shape[0] > 0:
        rows_to_remove2[i] = df2[df2[i] == -99999].shape[0]

rows_to_remove2 # empty dictionary, so no null values

# merging the two dataframes on common feature prospect_id
common_cols = df1.columns.intersection(df2.columns)
common_cols.tolist()

df = pd.merge(df1, df2, left_on=common_cols.tolist(), how='inner', right_on=common_cols.tolist())
df.shape
df.columns.value_counts().sort_values(ascending=False) # checking for duplicate columns

# also null values in the merged dataframe
df.isna().sum().sum()
(df == -99999).any().any() # checking for -99999 values
df.to_csv('final.csv', index=False) # saving the merged dataframe

df = pd.read_csv('final.csv')

# 2. Feature engineering 
# categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
cat_cols = cat_cols.drop('Approved_Flag').to_list()
target = ['Approved_Flag']
# Approved_Flag is the target variable

# numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_cols

# checking if cat_cols are related to the target variable or not using chi square test
for i in cat_cols:
    print(i,"=>", st.chi2_contingency(pd.crosstab(df[i], df['Approved_Flag'])).pvalue)

# all pvalues are less than 0.05, so all categorical variables are related to the target variable as per alternative hypothesis


# checking multicollinearity using VIF
# VIF > 6 indicates multicollinearity
num_cols = num_cols[num_cols != 'PROSPECTID'] # removing ProspectID from the list
num_cols = num_cols.to_list()
dropped = True
while dropped:
    dropped = False
    vif_data = [variance_inflation_factor(df[num_cols].values, i) for i in range(len(num_cols))]

    max_vif = max(vif_data)
    if max_vif > 6:
        indexx = vif_data.index(max_vif)
        print(f'Dropping {num_cols[indexx]} with VIF = {max_vif:.2f}')
        num_cols.pop(indexx)
        dropped = True

# now using anova test to check the relationship between numerical variables and target variable
anova_data = dict()
for i in num_cols:
    anova_data[i] = st.f_oneway(df[i][df['Approved_Flag'] == 'P1'], df[i][df['Approved_Flag'] == 'P2'], df[i][df['Approved_Flag'] == 'P3'], df[i][df['Approved_Flag'] == 'P4']).pvalue

useful_cols = list({i: anova_data[i] for i in anova_data if anova_data[i] <= 0.05}.keys()) 
# checking for pvalues > 0.05
# p-value ≤ 0.05: The variable likely has different means across the Approved_Flag categories, it may be useful for predicting or explaining group membership.

# p-value > 0.05: The variable likely has similar means across the groups, suggesting it might not be a strong discriminator for Approved_Flag.

new_df = df[useful_cols + cat_cols +  [target]].copy(deep=True) # creating a new dataframe with useful columns, categorical columns and target variable

new_df.columns.duplicated().any() # checking for duplicate columns
new_df.columns[new_df.columns.duplicated()]


# encoding categorical variables
for i in cat_cols:
    print(f'{i} : {df[i].nunique()}')
    print(df[i].value_counts())
    print('\n')

# only EDUCATION an be used for ordinal encoding as it has ordered values
# all other categorical variables are nominal and can be used for one hot encoding
# checking for unique values in EDUCATION
new_df['EDUCATION'].unique()
# ['12TH', 'GRADUATE', 'SSC', 'POST-GRADUATE', 'UNDER GRADUATE','OTHERS', 'PROFESSIONAL']
# EDUCATION has ordered values, so we can use ordinal encoding
# creating a mapping dictionary for ordinal encoding

education_map = {'12TH': 2, 'GRADUATE': 3, 'SSC': 1, 'POST-GRADUATE': 4, 'UNDER GRADUATE': 3, 'OTHERS': 1, 'PROFESSIONAL': 4}

# mapping the values
new_df['EDUCATION'] = new_df['EDUCATION'].map(education_map)
new_df['EDUCATION'].unique()
new_df['EDUCATION'].value_counts()

# one hot encoding for other categorical variables
df_encoded = pd.get_dummies(new_df, columns= [cols for cols in cat_cols if cols != 'EDUCATION'], dtype=int)

df_encoded.info()

df_encoded.to_csv('encoded.csv', index=False)

df_encoded = pd.read_csv('encoded.csv')

# 3. Model building
## 3.1 Random Forest Classifier

x = df_encoded.drop('Approved_Flag', axis=1)
y = df_encoded['Approved_Flag']
# splitting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')


## 3.2 xgboost


label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)
xgb_model.fit(x_train, y_train)
y_pred = xgb_model.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')



## 3.3 Support Vector Classifier
from sklearn.svm import SVC, LinearSVC
svc = SVC(kernel='rbf', random_state=42)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')

## 3.4 Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
x = df_encoded.drop('Approved_Flag', axis=1)
y = df_encoded['Approved_Flag']
# splitting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(x_train, y_train)
y_pred = gb.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')



## 3.5 adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

x = df_encoded.drop('Approved_Flag', axis=1)
y = df_encoded['Approved_Flag']
# splitting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=100,
    algorithm='SAMME',  # good for multiclass
    random_state=42
)
ada.fit(x_train, y_train)
y_pred = ada.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')


## 3.6 CatBoost
from catboost import CatBoostClassifier
cat = CatBoostClassifier(loss_function='MultiClass', eval_metric='Accuracy', iterations=200, random_state=42, verbose=100)

df = pd.read_csv('final.csv')

x = df[cat_cols + num_cols]
y = df[target]

x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.2, random_state=42)
x_train3, x_val3, y_train3, y_val3 = train_test_split(x_train2, y_train2, test_size=0.25, random_state=42, stratify=y_train2)
model = cat.fit(x_train3, y_train3, eval_set=(x_val3, y_val3), cat_features = cat_cols, plot=True)

y_pred2 = model.predict(x_test2)
print(f'Accuracy: {accuracy_score(y_test2, y_pred2):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test2, y_pred2)}')
print(f'Classification Report: \n{classification_report(y_test2, y_pred2)}')

# catboost has highest accuracy i.e., 0.99 and on that data we didn't did much preprocessing

train_preds = model.predict(x_train2)
test_preds = model.predict(x_test2)

print("Train Accuracy:", accuracy_score(y_train2, train_preds))
print("Test Accuracy:", accuracy_score(y_test2, test_preds))

# 4. getting feature importance for Catboost
importances = model.get_feature_importance()
feature_names = x_train3.columns

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

feat_imp_df.head()
# Credit_Score  Importance is 99%, all other features has less than 0.1% importance

pd.concat([x_train2['Credit_Score'], y_train2['Approved_Flag']], axis=1)

sns.boxplot(x=df['Credit_Score'], y=df['Approved_Flag'])
plt.show()
# clear separation of Credit_Score Range for each class.

print(pd.concat([df[df['Approved_Flag'] == 'P1']['Credit_Score'].describe(),
df[df['Approved_Flag'] == 'P2']['Credit_Score'].describe(),
df[df['Approved_Flag'] == 'P3']['Credit_Score'].describe(),
df[df['Approved_Flag'] == 'P4']['Credit_Score'].describe()], axis=1)
)


# lets try to train the model without Credit_Score
x_train2.drop('Credit_Score', axis=1, inplace=True)
x_test2.drop('Credit_Score', axis=1, inplace=True)

cat2 = CatBoostClassifier(loss_function='MultiClass', eval_metric='Accuracy', iterations=200, random_state=42, verbose=100)
model2 = cat2.fit(x_train2, y_train2, eval_set=(x_test2, y_test2), cat_features = cat_cols, plot=True)
y_pred2 = model2.predict(x_test2)
print(f'Accuracy: {accuracy_score(y_test2, y_pred2):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test2, y_pred2)}')
print(f'Classification Report: \n{classification_report(y_test2, y_pred2)}')


importances2 = model2.get_feature_importance()
feature_names2 = x_train2.columns

feat_imp_df2 = pd.DataFrame({
    'Feature': feature_names2,
    'Importance': importances2
}).sort_values(by='Importance', ascending=False)

feat_imp_df2[feat_imp_df2['Importance'] >=1]

# just modeling a xgboost model with Credit_Score
df_encoded = pd.read_csv('encoded.csv')
df_encoded['Credit_Score'] = df['Credit_Score']
x2 = df_encoded.drop(columns = ['Approved_Flag'], axis=1)
y2 = df['Approved_Flag']
label_encode = LabelEncoder()
y2_encoded = label_encode.fit_transform(y2)


x_train4, x_test4, y_train4, y_test4 = train_test_split(x2, y2_encoded, test_size=0.2, random_state=42)

xgb2 = xgb.XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)
xgb2.fit(x_train4, y_train4)
y_pred4 = xgb2.predict(x_test4)
print(f'Accuracy: {accuracy_score(y_test4, y_pred4):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test4, y_pred4)}')
print(f'Classification Report: \n{classification_report(y_test4, y_pred4)}')

# xgboost has highest accuracy i.e., 0.99 and on that data we didn't did much preprocessing

train_preds4 = xgb2.predict(x_train4)
test_preds4 = xgb2.predict(x_test4)

print("Train Accuracy:", accuracy_score(y_train4, train_preds4))
print("Test Accuracy:", accuracy_score(y_test4, test_preds4))


# applying anova for Credit_Score(numerical) and EDUCATION(categorical) variables
df['Credit_Score'].plot(kind='kde')

sm.qqplot(df['Credit_Score'], line ='45', fit=True)
plt.show()

groups = [group['Credit_Score'].values for name, group in df.groupby('Approved_Flag')]
kruskal_stat, p_val = st.kruskal(*groups)
print(f"Kruskal-Wallis H-statistic: {kruskal_stat}, p-value: {p_val}")

# Assuming df is your DataFrame
print(sp.posthoc_dunn(df, val_col='Credit_Score', group_col='Approved_Flag', p_adjust='bonferroni'))


# qqplot for Credit_Score to check normality
sns.displot(df['Credit_Score'], bins=30, kde=True)
plt.show()


## qq-plot
df['Approved_Flag'].value_counts()


## class_weight
from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight('balanced', classes = np.unique(y_encoded), y = y_encoded) 
class_weights = dict(zip(np.unique(y_encoded), weights))



## preparing a xgboost without credit score and using class_weight
x = df_encoded.drop('Approved_Flag', axis=1)
y = df_encoded['Approved_Flag']
# splitting the data into train and test
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)
x_train5, x_test5, y_train5, y_test5 = train_test_split(x, y_encoded, test_size=0.2, random_state=42)
sample_weights = np.array([class_weights[label] for label in y_train5])

xgb3 = xgb.XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss', random_state=42)
xgb3.fit(x_train5, y_train5, sample_weight = sample_weights, verbose=True)
y_pred5 = xgb3.predict(x_test5)
print(f'Accuracy: {accuracy_score(y_test5, y_pred5):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test5, y_pred5)}')
print(f'Classification Report: \n{classification_report(y_test5, y_pred5)}')



#-------------------------------------------------------------------------------------------

cat2 = CatBoostClassifier(loss_function='MultiClass', task_type='GPU',  eval_metric='Accuracy', iterations=250, random_state=42, verbose=100)
x = df_encoded.drop('Approved_Flag', axis=1)
y = df_encoded['Approved_Flag']
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.2, random_state=42)
x_train3, x_val3, y_train3, y_val3 = train_test_split(x_train2, y_train2, test_size=0.25, random_state=42, stratify=y_train2)
model2 = cat2.fit(x_train2, y_train2, eval_set=(x_test2, y_test2),  plot=True)
model2.get_all_params()
y_pred2 = model2.predict(x_test2)
print(f'Accuracy: {accuracy_score(y_test2, y_pred2):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test2, y_pred2)}')
print(f'Classification Report: \n{classification_report(y_test2, y_pred2)}')

# -----------------------------------------------------------------------------------

from catboost import CatBoostClassifier
x = df_encoded.drop('Approved_Flag', axis=1)
y = df_encoded['Approved_Flag']
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
weights = compute_class_weight('balanced', classes = np.unique(y), y = y) 
class_weights = dict(zip(np.unique(y), weights))
sample_weights = np.array([class_weights[label] for label in y_train2])
cat3 = CatBoostClassifier(loss_function='MultiClass', eval_metric='Accuracy', iterations=200, random_state=42, verbose=100, class_weights = weights)
model3 = cat3.fit(x_train2, y_train2, plot=True, eval_set=(x_test2, y_test2))
y_pred2 = model3.predict(x_test2)
print(f'Accuracy: {accuracy_score(y_test2, y_pred2):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test2, y_pred2)}')
print(f'Classification Report: \n{classification_report(y_test2, y_pred2)}')


# -----------------------------------------------------------------------------------

# using smote

df_encoded = pd.read_csv('encoded.csv')

x = df_encoded.drop('Approved_Flag', axis=1)
y = df_encoded['Approved_Flag']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
le.classes_, le.transform(le.classes_)
a_dict = dict(zip(le.transform(le.classes_).tolist(), le.classes_))

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42, stratify=y)

np.unique(y_train), np.unique(y_train, return_counts=True)[1]/y_train.shape[0] 
df_encoded['Approved_Flag'].value_counts(normalize=True)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

np.unique(y_train_smote), np.unique(y_train_smote, return_counts=True)[1]/y_train.shape[0] 
print("Before SMOTE:", dict(zip(*np.unique(y_train, return_counts=True))))
print("After SMOTE:", dict(zip(*np.unique(y_train_smote, return_counts=True))))



model3 = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    verbose=100,
    random_state=42
)

model3.fit(x_train_smote, y_train_smote, eval_set=(x_test, y_test), plot=True)
y_pred = model3.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')

xgb4 = xgb.XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)
xgb4.fit(x_train_smote, y_train_smote,  verbose=True)
y_pred = xgb4.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')


# hyperparameter tuning
param_grid = {
    'iterations': [300, 350, 400],                   # around base = 350
    'depth': [5, 6, 10],                              # base = 6
    'learning_rate': [0.15, 0.17, 0.2],              # base ≈ 0.173
    'l2_leaf_reg': [2, 3, 4],                        # base = 3
    'border_count': [128, 254],                     # base = 254 (high)
    'bagging_temperature': [0.5, 1, 2],              # base = 1
    'random_strength': [0.5, 1, 1.5],                # base = 1
    'bootstrap_type': ['Bayesian'],                 # base = Bayesian
    'boosting_type': ['Plain'],                     # base = Plain
    'score_function': ['Cosine'],                   # base = Cosine
    'grow_policy': ['SymmetricTree'],               # base = SymmetricTree
}
model4 = CatBoostClassifier(
    loss_function='MultiClass',
    verbose=0,
    random_state=42,
    task_type='GPU',          # enable GPU training
    devices='0',              # specify GPU device
    early_stopping_rounds=30  # stop early to save time
)

scorer = make_scorer(f1_score, average='macro')

search = RandomizedSearchCV(
    estimator=model4,
    param_distributions=param_grid,
    scoring=scorer,
    cv=3,
    n_iter=15,           # reduce number of iterations in RandomizedSearchCV
    random_state=42,
    verbose=2,
    n_jobs=1             # CatBoost GPU does not support parallel sklearn jobs; set to 1
)

search.fit(x_train, y_train)
# Best hyperparameters
print("Best Parameters:", search.best_params_)

# Best score achieved (on validation folds)
print("Best F1 Macro Score:", search.best_score_)


 
best_model = search.best_estimator_
y_pred = best_model.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')

best_model = search.best_estimator_
y_pred = best_model.predict(x_train)
print(f'Accuracy: {accuracy_score(y_train, y_pred):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_train, y_pred)}')
print(f'Classification Report: \n{classification_report(y_train, y_pred)}')


## training on smote data

search2 = RandomizedSearchCV(
    estimator=model4,
    param_distributions=param_grid,
    scoring=scorer,
    cv=3,
    n_iter=15,           # reduce number of iterations in RandomizedSearchCV
    random_state=42,
    verbose=2,
    n_jobs=1             # CatBoost GPU does not support parallel sklearn jobs; set to 1
)
search2.fit(x_train_smote, y_train_smote)
# Best hyperparameters
print("Best Parameters:", search2.best_params_)

# Best score achieved (on validation folds)
print("Best F1 Macro Score:", search2.best_score_)

best_model2 = search2.best_estimator_
y_pred = best_model2.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')

y_pred = best_model2.predict(x_train)
print(f'Accuracy: {accuracy_score(y_train, y_pred):.2f}')
print(f'Confusion Matrix: \n{confusion_matrix(y_train, y_pred)}')
print(f'Classification Report: \n{classification_report(y_train, y_pred)}')

# using SMOTE data causing overfitting as accuracy = 0.96 on train data and 0.76 on test data

# so using original data, which has 0.78 on train data and 0.81 on test data, is better

