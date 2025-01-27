import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Load and preprocess the dataset
df = pd.read_csv(r"C:\Users\ashis\OneDrive\Desktop\Python\Dataset\emails.csv")

# Remove duplicates and NaN values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Features and labels
X = df['text']
y = df['spam']

# Encode labels
l = LabelEncoder()
df['spam'] = l.fit_transform(df['spam'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data into feature vectors using CountVectorizer
cv = CountVectorizer()
X_train_vectorized = cv.fit_transform(X_train).toarray()  # Convert sparse to dense
X_test_vectorized = cv.transform(X_test).toarray()        # Convert sparse to dense

# Initialize and train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
accuracy = model.score(X_test_vectorized, y_test)
print(f"Model Accuracy: {accuracy}")



pickle.dump(model, open('email_spam_model.pkl', 'wb'))
pickle.dump(cv, open("count_vectorizer.pkl", "wb"))