
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

folder = Path(__file__).parent
file_path = folder / "processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv(
    file_path,
    header=None,
    names=columns,
    na_values="?"
)

# Assignment ke liye sirf 4 input attributes
features = ["age", "sex", "cp", "thalach"]

df = df[features + ["target"]].dropna()
df["target"] = (df["target"] > 0).astype(int)

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), SVC())
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

with open(folder / "heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Heart Disease model trained successfully!")
print(f"Accuracy: {accuracy:.2%}")
print("Input attributes:", features)
print("Saved model: heart_model.pkl")
