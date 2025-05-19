from utils import load_data, preprocess_data
from sklearn.ensemble import RandomForestClassifier
import joblib

df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Model accuracy: {score*100:.2f}%")

# Save model
joblib.dump(model, "model.pkl")
