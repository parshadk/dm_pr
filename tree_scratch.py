import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
# Show summary
print(X.describe())



def entropy(y):
    classes = np.unique(y)
    ent = 0
    for cls in classes:
        p = np.sum(y == cls) / len(y)
        ent -= p * np.log2(p)
    return ent
def gini(y):
    classes = np.unique(y)
    g = 1
    for cls in classes:
        p = np.sum(y == cls) / len(y)
        g -= p ** 2
    return g

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
def best_split(X, y, criterion='gini'):
    best_feat, best_thresh, best_gain = None, None, -1
    current_impurity = gini(y) if criterion == 'gini' else entropy(y)
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            left_idx = X[:, feature] <= t
            right_idx = ~left_idx

            if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                continue

            left_impurity = gini(y[left_idx]) if criterion == 'gini' else entropy(y[left_idx])
            right_impurity = gini(y[right_idx]) if criterion == 'gini' else entropy(y[right_idx])
            p = len(y[left_idx]) / len(y)

            gain = current_impurity - (p * left_impurity + (1 - p) * right_impurity)
            if gain > best_gain:
                best_feat, best_thresh, best_gain = feature, t, gain
    return best_feat, best_thresh
def build_tree(X, y, depth=0, max_depth=3, criterion='gini'):
    if len(set(y)) == 1 or depth == max_depth:
        return Node(value=np.bincount(y).argmax())

    feat, thresh = best_split(X, y, criterion)
    if feat is None:
        return Node(value=np.bincount(y).argmax())

    left_idx = X[:, feat] <= thresh
    right_idx = ~left_idx

    left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth, criterion)
    right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth, criterion)

    return Node(feature=feat, threshold=thresh, left=left, right=right)
def predict_one(x, node):
    while node.value is None:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value
def predict(X, tree):
    return [predict_one(x, tree) for x in X]



X_np = X.values
X_train, X_test, y_train, y_test = train_test_split(X_np, y, test_size=0.2, random_state=42)
# Train
tree = build_tree(X_train, y_train.values, max_depth=4, criterion='entropy')
y_pred = predict(X_test, tree)
# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.show()



sample = np.array([[5.1, 3.5, 1.5, 0.2]])
custom_pred = predict(sample, tree)
print("Custom prediction:", iris.target_names[custom_pred[0]])


for d in [1, 2, 3, 4, 5]:
    t = build_tree(X_train, y_train.values, max_depth=d)
    p = predict(X_test, t)
    print(f"Depth {d} Accuracy: {accuracy_score(y_test, p):.2f}")
