# ================================
# SENTIMENT ANALYSIS ON TWITTER DATASET (FULL CODE)
# ================================

# Install (run once in terminal if needed)
# pip install pandas scikit-learn nltk

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None
)

df.columns = ["sentiment", "id", "date", "query", "user", "text"]
df = df[["sentiment", "text"]]

# Convert labels
df["sentiment"] = df["sentiment"].replace({0: "negative", 4: "positive"})

# -------------------------------
# 2. Preprocessing
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

# -------------------------------
# 3. Sampling (for faster training)
# -------------------------------
df_sample = df.sample(50000, random_state=42)

X = df_sample["clean_text"]
y = df_sample["sentiment"]

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# 6. Train Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# -------------------------------
# 7. Evaluation
# -------------------------------
y_pred = model.predict(X_test_vec)

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------------
# 8. Prediction Function
# -------------------------------
def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

# -------------------------------
# 9. Test Examples
# -------------------------------
print("\nSample Predictions:")
print("1:", predict_sentiment("I love this phone so much!"))
print("2:", predict_sentiment("Worst experience ever"))
print("3:", predict_sentiment("It is okay, not great"))

# ================================
# END
# ================================