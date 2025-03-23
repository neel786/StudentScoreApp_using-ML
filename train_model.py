import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
data = pd.read_csv("data/StudentsPerformance.csv")

# Select features and target
features = ['test preparation course', 'parental level of education', 'reading score']
X = data[features]
y = data['math score']

# Preprocessing: One-hot encode categorical features
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['test preparation course', 'parental level of education'])
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', column_transformer),
    ('regressor', RandomForestRegressor(
        n_estimators=50,  # Reduce the number of trees
        max_depth=10,     # Limit the depth of trees to prevent overfitting
        min_samples_split=5,  # Require at least 5 samples to split a node
        random_state=42
    ))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
print("Training score:", model.score(X_train, y_train))
print("Test score:", model.score(X_test, y_test))

# Save the model
with open("model/exam_score_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")