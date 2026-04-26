import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
data = pd.read_csv("heart.csv")  # make sure file is here

# Encode manually (same as before)
data['Sex'] = data['Sex'].map({'M':1, 'F':0})
data['ChestPainType'] = data['ChestPainType'].map({'ATA':0, 'NAP':1, 'ASY':2, 'TA':3})
data['RestingECG'] = data['RestingECG'].map({'Normal':0, 'ST':1, 'LVH':2})
data['ExerciseAngina'] = data['ExerciseAngina'].map({'N':0, 'Y':1})
data['ST_Slope'] = data['ST_Slope'].map({'Up':0, 'Flat':1, 'Down':2})

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

model = RandomForestClassifier()
model.fit(X, y)

# Save locally (IMPORTANT)
joblib.dump(model, "model_gb.pkl")

print("Model saved successfully!")