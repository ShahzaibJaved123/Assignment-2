import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
data_file = "Covid Data.csv"
df = pd.read_csv(data_file)

# Preprocessing
# Replace '97' with NaN for missing data
missing_value = 97
df.replace(missing_value, np.nan, inplace=True)

# Fill missing values with mode for categorical and median for numerical columns
for col in df.columns:
    if df[col].dtype == 'object' or len(df[col].unique()) < 10:
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
for col in ['SEX', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION',
            'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Feature Selection
X = df.drop(columns=['DATE_DIED', 'CLASIFFICATION_FINAL'])
y = df['CLASIFFICATION_FINAL']

# Splitting dataset for classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Visualization of Clustering
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()

# 2. Naive Bayes Classification
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred = naive_bayes.predict(X_test)

# Performance Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(ticks=np.arange(conf_matrix.shape[1]), labels=np.arange(conf_matrix.shape[1]))
plt.yticks(ticks=np.arange(conf_matrix.shape[0]), labels=np.arange(conf_matrix.shape[0]))
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')

# Annotate the matrix
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')

plt.show()

# Print Performance Metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# Check the number of classes and instances
num_classes = df['CLASIFFICATION_FINAL'].nunique()
num_instances = df.shape[0]
print(f"Number of classes: {num_classes}")
print(f"Number of instances: {num_instances}")