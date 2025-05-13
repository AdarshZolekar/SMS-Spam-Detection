import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

df = pd.read_csv('dataset/spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def clean_text(msg):
    msg = msg.lower()
    msg = ''.join([c for c in msg if c not in string.punctuation])
    msg = ' '.join([word for word in msg.split() if word not in stopwords.words('english')])
    return msg

df['message'] = df['message'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

msg = ["Congratulations! You've won a free gift card."]
msg_clean = vectorizer.transform([clean_text(msg[0])])
print("Prediction (1=spam, 0=ham):", model.predict(msg_clean)[0])
