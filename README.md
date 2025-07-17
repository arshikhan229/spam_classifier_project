# 📧 Spam Classifier using Python & Scikit-learn

This project is a simple **SMS spam classifier** that uses **machine learning (Naive Bayes)** to detect whether a message is "ham" (not spam) or "spam". It is built using Python, pandas, scikit-learn, and TfidfVectorizer.

---

## 🗂️ Dataset

The dataset is a collection of 5573 SMS messages labeled as `ham` or `spam`. You can find or download it from sources like [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).

Format example:
ham,I'm leaving my house now...
spam,You have won $1000! Call now!

---

## ⚙️ Installation & Setup

1. Clone the repository:
```bash

git clone https://github.com/yourusername/spam_classifier_project.git
cd spam_classifier_project
Create a virtual environment and activate it:
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
Install required libraries:
pandas
scikit-learn
joblib

🚀 Run the Classifier
To train the model and test predictions:
python spam_classifier.py
It will:

Load and clean the dataset

Train a Naive Bayes classifier

Predict test accuracy

Predict whether a new message is spam or ham

🧠 Example Usage in Code
You can test your own messages:
def classify_message(msg):
    msg_vec = vectorizer.transform([msg])
    result = model.predict(msg_vec)
    return "spam" if result[0] == 1 else "ham"

# Try some predictions
print(classify_message("Congratulations! You've won a free ticket. Call now!"))
print(classify_message("Hey, are you coming to the meeting today?"))
💾 Save and Load Trained Model
Save:
import joblib
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
Load later:
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
📊 Accuracy
Expected accuracy with a proper train/test split is typically around 95% or higher, depending on preprocessing and vectorization.

🧪 Tech Stack
Python 3

Pandas

Scikit-learn

TfidfVectorizer

Multinomial Naive Bayes

📄 License
This project is open-source and available under the MIT License.

