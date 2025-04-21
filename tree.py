from sklearn.datasets import load_iris
import pandas as pd
# Load iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].apply(lambda x: iris.target_names[x])
# Summary statistics
print(df.describe())
print(df['species'].value_counts())


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# Features and labels
X = df[iris.feature_names]
Y = df['target']
# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
# Initialize model (You can change criterion to 'entropy')
clf = DecisionTreeClassifier(criterion='gini', random_state=1)
clf.fit(X_train, Y_train)


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(12, 6))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Structure")
plt.show()


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
Y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy:", accuracy)
# Confusion matrix
cm = confusion_matrix(Y_test,Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

import numpy as np
# Example: sepal length = 5.1, sepal width = 3.5, petal length = 1.5, petal width = 0.2
custom_input = np.array([[5.1, 3.5, 1.5, 0.2]])
prediction = clf.predict(custom_input)
predicted_class = iris.target_names[prediction[0]]
print("Predicted class:", predicted_class)


# Train with max_depth=3
clf_pruned = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1)
clf_pruned.fit(X_train, Y_train)
Y_pred_pruned = clf_pruned.predict(X_test)
# Accuracy comparison
acc_pruned = accuracy_score(Y_test, Y_pred_pruned)
print("Original Accuracy:", accuracy)
print("Pruned Tree Accuracy (max_depth=3):", acc_pruned)
# Optinal: Plot pruned tree
plt.figure(figsize=(12, 6))
plot_tree(clf_pruned, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Pruned Decision Tree")
plt.show()

