
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

os.makedirs("outputs", exist_ok=True)

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("Dataset Shape:", X.shape)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nModel Evaluation")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1]
})

metrics.to_csv("outputs/evaluation_metrics.csv", index=False)

plt.figure(figsize=(7, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.title("Iris Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.bar(metrics["Metric"], metrics["Score"])
plt.ylim(0, 1.05)
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("outputs/accuracy_comparison.png", dpi=300)
plt.close()

print("\nAll evaluation outputs saved successfully.")
