import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# We use latin-1 to avoid errors with special characters like £
df = pd.read_csv(r'C:\Users\devis\Documents\Osasis\spam.csv', encoding='latin-1')
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

def check_emails(messages_list):
    transformed_input = tfidf.transform(messages_list)
    raw_predictions = model.predict(transformed_input)
    print("\n--- SPAM DETECTOR RESULTS ---")
    for msg, numeric_label in zip(messages_list, raw_predictions):
        text_label = "SPAM" if numeric_label == 1 else "HAM"
        print(f"[{text_label}] -> {msg}")

my_test_list = [
    "Hey, are you coming to the party tonight?",
    "Your package has been delivered. Please check your porch.",
    "WINNER! Claim your $500 Amazon gift card by clicking this link.",
    "I'll be a bit late, don't wait for me."
]
check_emails(my_test_list)