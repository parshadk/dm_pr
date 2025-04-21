import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])
# Basic preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation/numbers
    return text.split()
# Apply preprocessing
df['tokens'] = df['message'].apply(preprocess)
# Convert labels to binary
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})


X_train, X_test, y_train, y_test = train_test_split(df['tokens'], df['label_num'], test_size=0.2, random_state=42)


class NaiveBayes:
    def fit(self, X, y):
        self.vocab = set()
        self.word_counts = {0: defaultdict(int), 1: defaultdict(int)}
        self.class_counts = {0: 0, 1: 0}
        self.total_words = {0: 0, 1: 0}
        for tokens, label in zip(X, y):
            self.class_counts[label] += 1
            for word in tokens:
                self.vocab.add(word)
                self.word_counts[label][word] += 1
                self.total_words[label] += 1
        self.class_probs = {
            0: self.class_counts[0] / len(y),
            1: self.class_counts[1] / len(y)
        }
    def predict(self, X):
        return [self._predict_one(x) for x in X]
    def _predict_one(self, tokens):
        log_prob = {0: np.log(self.class_probs[0]), 1: np.log(self.class_probs[1])}
        vocab_size = len(self.vocab)
        for c in [0, 1]:
            for word in tokens:
                word_freq = self.word_counts[c][word]
                prob = (word_freq + 1) / (self.total_words[c] + vocab_size)
                log_prob[c] += np.log(prob)

        return 1 if log_prob[1] > log_prob[0] else 0


# Train
nb = NaiveBayes()
nb.fit(X_train, y_train)
# Predict
y_pred = nb.predict(X_test.tolist())
# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))



def preprocess_message(msg):
    return re.sub(r'[^a-z\s]', '', msg.lower()).split()
new_msg = "Win a free iPhone now!"
processed = preprocess_message(new_msg)
prediction = nb._predict_one(processed)
print(f"'{new_msg}' is classified as: {'SPAM' if prediction else 'HAM'}")



cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
