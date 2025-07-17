import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

import pandas as pd

# Read the dataset and skip the first row (header already present in file)
data = pd.read_csv("dataset.csv", encoding="ISO-8859-1", usecols=[0, 1], names=['label', 'message'], skiprows=1)

# Preview and verify
print(data.info())
print(data.head())
# Check for any missing values
print(data.isnull().sum())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Example: Predict a new message
new_message = ["Congratulations! You've won a free ticket to Bahamas. Call now!"]

# Vectorize the message using the same vectorizer
new_message_vec = vectorizer.transform(new_message)

# Predict
prediction = model.predict(new_message_vec)

# Interpret the result
label = "spam" if prediction[0] == 1 else "ham"
print(f"Prediction: {label}")
def classify_message(msg):
    msg_vec = vectorizer.transform([msg])
    result = model.predict(msg_vec)
    return "spam" if result[0] == 1 else "ham"

# Example usage
print(classify_message("Can we meet at 6pm at the cafe?"))
print(classify_message("WINNER!! Click here to claim your $1000 prize"))

