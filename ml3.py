# ============================================
# LAB 03 – kNN Classification (TED Main v2)
# Subject: 22AIE213
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import minkowski
from sklearn.metrics import confusion_matrix

# -------------------------------------------------
# A1: Dot Product & Euclidean Norm
# -------------------------------------------------

def dot_product(A, B):
    return sum(a * b for a, b in zip(A, B))

def euclidean_norm(A):
    return (sum(a * a for a in A)) ** 0.5

# -------------------------------------------------
# A2: Mean & Standard Deviation
# -------------------------------------------------

def mean_vector(X):
    return np.mean(X, axis=0)

def std_vector(X):
    return np.std(X, axis=0)

def class_stats(X, y, label):
    data = X[y == label]
    return mean_vector(data), std_vector(data)

# -------------------------------------------------
# A4: Minkowski Distance
# -------------------------------------------------

def minkowski_distance(v1, v2, p):
    return (sum(abs(a - b) ** p for a, b in zip(v1, v2))) ** (1 / p)

# -------------------------------------------------
# A10: Own kNN Implementation
# -------------------------------------------------

def own_knn_predict(X_train, y_train, test_vec, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_norm(X_train[i] - test_vec)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    labels = [label for _, label in distances[:k]]
    return max(set(labels), key=labels.count)

# -------------------------------------------------
# A13: Performance Metrics
# -------------------------------------------------

def performance_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return accuracy, precision, recall, f1

# -------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------

# Load dataset
df = pd.read_csv(
    r"C:\Users\cbswa\Downloads\sran\OneDrive\Documents\OneDrive\Desktop\ted_main_v2.csv"
)

# Convert views from text (with commas) to numeric
df["views"] = df["views"].astype(str).str.replace(",", "")
df["views"] = pd.to_numeric(df["views"], errors="coerce")

# Drop invalid rows
df = df.dropna(subset=["views", "about_talk"])

# Create binary confidence label using median views
median_views = df["views"].median()
df["confidence"] = (df["views"] >= median_views).astype(int)

# Text and labels
X_text = df["about_talk"]
y = df["confidence"].values

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
X = vectorizer.fit_transform(X_text).toarray()

# ---------------- A1 ----------------
A = X[0]
B = X[1]

print("\nA1: Vector Operations")
print("Dot Product (Manual):", dot_product(A, B))
print("Dot Product (NumPy):", np.dot(A, B))
print("Euclidean Norm (Manual):", euclidean_norm(A))
print("Euclidean Norm (NumPy):", np.linalg.norm(A))

# ---------------- A2 ----------------
mean0, std0 = class_stats(X, y, 0)
mean1, std1 = class_stats(X, y, 1)

interclass_distance = np.linalg.norm(mean0 - mean1)
print("\nA2: Inter-class Distance:", interclass_distance)

# ---------------- A3 ----------------
plt.hist(X[:, 0], bins=10)
plt.title("Histogram of Feature 1")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.show()

# ---------------- A4 & A5 ----------------
distances = []
for p in range(1, 11):
    distances.append(minkowski_distance(A, B, p))

plt.plot(range(1, 11), distances, marker="o")
plt.xlabel("p value")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance vs p")
plt.show()

print("\nA5: SciPy Minkowski (p=3):", minkowski(A, B, 3))

# ---------------- A6 ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------- A7 & A8 ----------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("\nA7 & A8: Test Accuracy:", knn.score(X_test, y_test))

# ---------------- A9 ----------------
predictions = knn.predict(X_test)

# ---------------- A10 ----------------
own_predictions = np.array(
    [own_knn_predict(X_train, y_train, x, 3) for x in X_test]
)
print("\nA10: Own kNN Accuracy:", np.mean(own_predictions == y_test))

# ---------------- A11 ----------------
k_vals = range(1, 12)
acc = []

for k in k_vals:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc.append(model.score(X_test, y_test))

plt.plot(k_vals, acc, marker="s")
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("k vs Accuracy")
plt.show()

# ---------------- A12 & A13 ----------------
cm = confusion_matrix(y_test, predictions)
accuracy, precision, recall, f1 = performance_metrics(cm)

print("\nConfusion Matrix:\n", cm)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")