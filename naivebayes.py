import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
# Download NLTK resources
nltk.download('stopwords')
# Load the dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
# Text preprocessing
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
df['clean_message'] = df['message'].apply(preprocess)
print(df.head())



from sklearn.feature_extraction.text import CountVectorizer
# Vectorize messages
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_message'])
Y = df['label']
# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, Y_train)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
Y_pred = model.predict(X_test)
print("AccuracY:", accuracy_score(Y_test, Y_pred))
print("Precision:", precision_score(Y_test, Y_pred, pos_label='spam'))
print("Recall:", recall_score(Y_test, Y_pred, pos_label='spam'))
print("F1 Score:", f1_score(Y_test, Y_pred, pos_label='spam'))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))


new_message = ["Win a free iPhone now!"]
new_message_clean = [preprocess(m) for m in new_message]
new_message_vec = vectorizer.transform(new_message_clean)
prediction = model.predict(new_message_vec)
print("Prediction for new message:", prediction[0])



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(Y_test, Y_pred, labels=['ham', 'spam'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
