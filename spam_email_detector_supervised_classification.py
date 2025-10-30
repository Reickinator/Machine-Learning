import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Step-2 Prepare the Data
# data = {
#     'Email': [
#         "Win a free iPhone now",
#         "Meeting tomorrow at 10am",
#         "Lowest price on meds, buy now",
#         "Project deadline extended",
#         "Congratulations! You won a lottery",
#         "Can you review the report?",
#         "Earn money fast online",
#         "Lunch at noon?"
#     ],
#     'Label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
# }

# Read the CSV file
df = pd.read_csv('/Users/kennyreick/PycharmProjects/MachineLearning/spam_sms_message_data.csv')

# The first column contains the labels (ham/spam)
# The second column contains the text messages
# The remaining columns are empty, so we'll drop them

# Keep only the first two columns and rename them
df = df.iloc[:, :2]
df.columns = ['Label', 'Message']

# Remove any rows with missing values
df = df.dropna()

# Split into X (messages) and Y (labels)
X = df['Message'].values  # Text messages
Y = df['Label'].values    # Labels (ham/spam)

# Optional: Convert labels to binary (0 for ham, 1 for spam)
Y = (Y == 'spam').astype(int)

# df = pd.DataFrame(data)

# # Step-3 Split the Data
# X = df['Email']
# y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42
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
new_email = ["Hello"]
new_vector = vectorizer.transform(new_email)
prediction = model.predict(new_vector)

print("Spam" if prediction[0] == 1 else "Not Spam")

