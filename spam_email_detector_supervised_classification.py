import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step-2 Prepare the Data
data = {
    'Email': [
        "Win a free iPhone now",
        "Meeting tomorrow at 10am",
        "Lowest price on meds, buy now",
        "Project deadline extended",
        "Congratulations! You won a lottery",
        "Can you review the report?",
        "Earn money fast online",
        "Lunch at noon?"
    ],
    'Label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# Step-3 Split the Data
X = df['Email']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step-4 Split the Data
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Step-5 Train the Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Step-6 Make Predictions
y_pred = model.predict(X_test_vectors)

# Step-7 Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step-8 Test on a New Email
new_email = ["Report Friday"]
new_vector = vectorizer.transform(new_email)
prediction = model.predict(new_vector)

print("Spam" if prediction[0] == 1 else "Not Spam")

