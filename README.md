'''import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("card.zip")

# Drop the 'ID' column after inspecting the data
df.head(3)
df = df.drop("ID", axis=1)

# Separate the target variable and features
y = df["default.payment.next.month"]
x = df.drop(columns=["default.payment.next.month"])

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.77, random_state=6)

# Create and train the logistic regression model
log = LogisticRegression(max_iter=1000)
model = log.fit(x_train, y_train)

# Evaluate the model's accuracy using the test set
score = model.score(x_test, y_test)
print("Model accuracy score:", score)

# Make a prediction for a specific data point
denemex = np.array(x.iloc[1903]).reshape(1, -1)
prediction = model.predict(denemex)

# Show the actual result
print("Predicted result:", prediction)
print("Actual value:", y.iloc[1903])'''
