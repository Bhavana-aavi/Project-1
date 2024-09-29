# Project-1
#Music genre PCA and logistic regression project

import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df= pd.read_csv("C:/Users/drbha/Downloads/project-files-music-genre-classification-with-pca (1)/music_dataset_mod.csv")
df.info()
print(df.head())
missing_values= df.isnull().sum()
print(missing_values)
unique_genres= df['Genre'].unique()
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Genre')
plt.xticks(rotation=90)
plt.title('Distribution of Music Genres')
plt.show()
df_cleaned = df.dropna(subset=['Genre'])
X = df_cleaned.drop(columns= ['Genre'])
y = df_cleaned['Genre']
y.unique()
label_encoder= LabelEncoder()
y_encoded= labelencoder.fit_transform(y)
df_encoded = X.copy()
df_encoded['Genre_encoded'] = y_encoded
df_encoded.columns
corr_matrix = df_encoded.corr()
corr_matrix
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot= True, cmap= 'coolwarm', fmt= '.2f')
plt.title('Correlation Matrix heatmap')
plt.show()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)
pca= PCA()
X_pca = pca.fit_transform(X_scaled)
len(X_pca)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
plt.figure(figsize=(10,8))
plt.plot(cumulative_variance, marker='o', linestyle= '--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Principal Components')
plt.grid(True)
plt.axhline(y=0.80, color='r', linestyle='--')
plt.show()
optimal_components = np.argmax(cumulative_variance >= 0.80)+1
pca_optimal= PCA(n_components = optimal_components)
pca_optimal
X_pca_optimal = pca.fit_transform(X_scaled)
X_pca_optimal.shape
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca_optimal, y_encoded, test_size= 0.3, random_state= 42)
log_reg_pca = LogisticRegression(max_iter= 10000)
log_reg_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = log_reg_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
print(f"Accuracy with PCA : {accuracy_pca:.2f}" )
report_pca = classification_report(y_test_pca, y_pred_pca)
print('Classification report (PCA):')
print(report_pca)
X_train_orig,X_test_orig, y_train_orig, y_test_orig = train_test_split(X_scaled, y_encoded, test_size= 0.3, random_state= 42)
log_reg_orig = LogisticRegression(max_iter= 10000)
log_reg_orig.fit(X_train_orig, y_train_orig)
y_pred_orig = log_reg_orig.predict(X_test_orig)
accuracy_orig= accuracy_score(y_test_orig, y_pred_orig)
print(f'Accuracy (Original): {accuracy: .2f}')
report_orig = classification_report(y_test_orig, y_pred_orig)
print("Classification report (Original):")
print(report_orig)
unknown_genre = df[df['Genre'].isnull()].copy()
unknown_genre.head()
unknown_genre.loc[:,'Genre_encoded']= 0
X_unknown = unknown_genre.drop(columns= ['Genre'])
X_unknown_scaled = scaler.transform(X_unknown)
X_unknown_scaled.shape
X_unknown_pca = pca.fit_transform(X_unknown_scaled)
X_unknown_pca.shape
y_pred_unknown = log_reg_pca.predict(X_unknown_pca)
y = df_cleaned['Genre']

label_encoder= LabelEncoder()
label_encoder.fit(y)
predicted_genres= label_encoder.inverse_transform(y_pred_unknown)
print("Predicted genres:", predicted_genres)
predicted_genres.shape
df['Genre']
missing_genre_indices = df[df['Genre'].isna()].index
print(f"Number of missing genres: {len(missing_genre_indices)}")
print(f"Number of predicted genres: {len(predicted_genres)}")
if len(missing_genre_indices) == len(predicted_genres):
    # Assign predicted genre labels to the corresponding rows in the original DataFrame
    df.loc[missing_genre_indices, 'Genre'] = predicted_genres
    print(df['Genre'])
else:
    print("Mismatch in lengths: Cannot assign predicted genres.")
    # Check for any remaining missing values
missing_values_after = df['Genre'].isna().sum()
print(f"Missing values after assignment: {missing_values_after}")
df.info()
df.head()

