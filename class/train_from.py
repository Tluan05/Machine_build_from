
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model.logistics_model import Logistics_From_Scratch
from model.KNN_model import KNN

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"D:\Machine_build_from\data\data_spam_feature.csv", encoding='latin-1') 

data['label_num'] = data['label'].map({'Ham': 0, 'Spam': 1})

X = data['text']
y = data['label_num']

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa văn bản
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = KNN()

model.fit(X_train_vec, y_train)

predict = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predict)
precision = precision_score(y_test, predict)
recall = recall_score(y_test, predict)
f1 = f1_score(y_test, predict)
print (f"Accuracy: {accuracy}")
print (f"Precision: {precision}")
print (f"Recall: {recall}")
print (f"F1: {f1}")

cm = confusion_matrix(y_test, predict)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
