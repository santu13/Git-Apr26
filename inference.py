load pkl ml mdeol and make inference
import joblib
import numpy as np
from train import X_test

# Load the trained model
model = joblib.load("trained_model.pkl")

# Make predictions
predictions = model.predict(X_test)