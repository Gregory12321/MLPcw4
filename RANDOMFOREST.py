import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report

df = pd.read_excel("data_with_status_doc_prob.xlsx")

def parse_seat_counts(val):
    if pd.isnull(val):
        return {}
    try:
        return ast.literal_eval(val)
    except Exception:
        return {}

df["seat_counts_parsed"] = df["seat_counts"].apply(parse_seat_counts)
df["Labour_seats"] = df["seat_counts_parsed"].apply(lambda d: d.get("Labour", 0))
df["Conservative_seats"] = df["seat_counts_parsed"].apply(lambda d: d.get("Conservative", 0))
df["LibDem_seats"] = df["seat_counts_parsed"].apply(lambda d: d.get("Liberal Democrat", 0))

df_model = df[df["progress_status"].isin([1, 2])].copy()

def parse_doc_prob(val):
    if pd.isnull(val):
        return []
    try:
        val = val.strip("[]")
        parts = val.split()
        return [float(x) for x in parts]
    except Exception:
        return []



print("Target distribution:\n", df_model["progress_status"].value_counts())

features = [
    "Labour_seats", 
    "Conservative_seats", 
    "LibDem_seats", 
    "sponsor_party"]
target = "progress_status"

X = df_model[features]
y = df_model[target]

X_encoded = pd.get_dummies(X, columns=["sponsor_party"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=32)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print("\nTraining Accuracy: {:.2f}%".format(accuracy_score(y_train, y_pred_train) * 100))
print("Test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_test) * 100))
y_prob_test = model.predict_proba(X_test)
try:
    ll = log_loss(y_test, y_prob_test)
except Exception:
    ll = np.nan
print("Test Log Loss: {:.4f}".format(ll))
print("\nClassification Report (Test Data):\n", classification_report(y_test, y_pred_test))

y_prob_all = model.predict_proba(X_encoded)
class_idx = list(model.classes_).index(2)
all_prob = y_prob_all[:, class_idx]

results_df = pd.DataFrame({
    "bill_id": df_model["bill_id"],
    "RF_Probability": all_prob
})
results_df = results_df.drop_duplicates(subset="bill_id", keep="first")
results_df.to_csv("predictions_rf.csv", index=False)
print("\nPredictions for all bills saved to 'predictions_rf.csv'.")
