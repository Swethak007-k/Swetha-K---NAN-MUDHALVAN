import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic transaction data
np.random.seed(42)
num_transactions = 5000
data = {
    "amount": np.random.uniform(1, 1000, num_transactions),
    "time": np.random.randint(0, 24, num_transactions),  # Transaction hour
    "location": np.random.randint(0, 50, num_transactions),  # Randomized location ID
    "fraudulent": np.random.choice([0, 1], num_transactions, p=[0.97, 0.03])  # 3% fraud rate
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split data into training and test sets
X = df[["amount", "time", "location"]]
y = df["fraudulent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Example usage
sample_transaction = np.array([[200, 10, 15]])  # A sample transaction (amount=200, time=10 AM, location ID=15)
prediction = model.predict(sample_transaction)
print("Fraudulent" if prediction[0] else "Legitimate")
