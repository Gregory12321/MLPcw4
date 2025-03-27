import pandas as pd
import numpy as np
import ast
import glob
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def preprocess_data(file):

    df = pd.read_excel(file)
    
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
    
    def parse_doc_prob(val):
        if pd.isnull(val):
            return []
        try:
            val = val.strip("[]")
            parts = val.split()
            return [float(x) for x in parts]
        except Exception:
            return []
    
    df["doc_prob_parsed"] = df["doc_prob"].apply(parse_doc_prob)
    doc_prob_df = pd.DataFrame(df["doc_prob_parsed"].tolist(), index=df.index)
    doc_prob_df.columns = ["doc_prob1", "doc_prob2", "doc_prob3", "doc_prob4"]
    df = pd.concat([df, doc_prob_df], axis=1)
    
    df_model = df[df["progress_status"].isin([1, 2])].copy()
    return df_model

def prepare_features(df, feature_cols, target_col):

    X = df[feature_cols]
    y = df[target_col]
    X_encoded = pd.get_dummies(X, columns=["sponsor_party"], drop_first=True)
    return X_encoded, y

def get_probabilities(model, X):

    y_prob = model.predict_proba(X)
    class_idx = list(model.classes_).index(2)
    return y_prob[:, class_idx]

def process_doc_probs(val):


    if isinstance(val, str):
        val = val.strip('[]')
        parts = val.split()
        try:
            nums = [float(x) for x in parts]
            return max(nums)
        except:
            return np.nan
    return val


train_df = preprocess_data("finalTraining.xlsx")
test_df = preprocess_data("finalTesting.xlsx")

feature_cols = ["Labour_seats", "Conservative_seats", "LibDem_seats", "sponsor_party", 
                "doc_prob1", "doc_prob2", "doc_prob3", "doc_prob4"]
target_col = "progress_status"

X_train, y_train = prepare_features(train_df, feature_cols, target_col)
X_test, y_test = prepare_features(test_df, feature_cols, target_col)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

lr_model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.0, 
                              random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=32)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)
print("Logistic Regression:")
print("   Training Accuracy: {:.2f}%".format(accuracy_score(y_train, lr_train_pred)*100))
print("   Test Accuracy: {:.2f}%".format(accuracy_score(y_test, lr_test_pred)*100))
print()

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)
print("Random Forest:")
print("   Training Accuracy: {:.2f}%".format(accuracy_score(y_train, rf_train_pred)*100))
print("   Test Accuracy: {:.2f}%".format(accuracy_score(y_test, rf_test_pred)*100))
print()

gb_train_pred = gb_model.predict(X_train)
gb_test_pred = gb_model.predict(X_test)
print("Gradient Boosting:")
print("   Training Accuracy: {:.2f}%".format(accuracy_score(y_train, gb_train_pred)*100))
print("   Test Accuracy: {:.2f}%".format(accuracy_score(y_test, gb_test_pred)*100))
print()

train_lr_prob = get_probabilities(lr_model, X_train)
train_rf_prob = get_probabilities(rf_model, X_train)
train_gb_prob = get_probabilities(gb_model, X_train)

train_predictions = pd.DataFrame({
    "bill_id": train_df["bill_id"].values,
    "LogReg_Probability": train_lr_prob,
    "RF_Probability": train_rf_prob,
    "GB_Probability": train_gb_prob
}).drop_duplicates(subset="bill_id", keep="first")
train_predictions.to_csv("predictions_base_train.csv", index=False)

test_lr_prob = get_probabilities(lr_model, X_test)
test_rf_prob = get_probabilities(rf_model, X_test)
test_gb_prob = get_probabilities(gb_model, X_test)

test_predictions = pd.DataFrame({
    "bill_id": test_df["bill_id"].values,
    "LogReg_Probability": test_lr_prob,
    "RF_Probability": test_rf_prob,
    "GB_Probability": test_gb_prob
}).drop_duplicates(subset="bill_id", keep="first")
test_predictions.to_csv("predictions_base_test.csv", index=False)

full_text_train = pd.read_excel("fulLTextTraining.xlsx")
full_text_train.rename(columns={"id": "bill_id", "doc_probs": "doc_probs_2"}, inplace=True)
merged_train = pd.merge(train_predictions, full_text_train[["bill_id", "doc_probs_2"]], 
                        on="bill_id", how="left")
final_train = pd.merge(train_df[["bill_id", "progress_status"]], merged_train, 
                       on="bill_id", how="left")
final_train["doc_probs_2"] = final_train["doc_probs_2"].apply(process_doc_probs)
final_train.to_excel("merged_probabilities_train.xlsx", index=False)

full_text_test = pd.read_excel("fullTextTesting.xlsx")
full_text_test.rename(columns={"id": "bill_id", "doc_probs": "doc_probs_2"}, inplace=True)
merged_test = pd.merge(test_predictions, full_text_test[["bill_id", "doc_probs_2"]], 
                       on="bill_id", how="left")
final_test = pd.merge(test_df[["bill_id", "progress_status"]], merged_test, 
                      on="bill_id", how="left")
final_test["doc_probs_2"] = final_test["doc_probs_2"].apply(process_doc_probs)
final_test.to_excel("merged_probabilities_test.xlsx", index=False)

meta_features = ["LogReg_Probability", "RF_Probability", "GB_Probability", "doc_probs_2"]
X_meta_train = final_train[meta_features]
y_meta_train = final_train["progress_status"]

meta_model = LogisticRegression(random_state=42, max_iter=10000)
meta_model.fit(X_meta_train, y_meta_train)

train_meta_pred = meta_model.predict(X_meta_train)
print("Final Meta-Model (on Training Data):")
print("   Accuracy: {:.2f}%".format(accuracy_score(y_meta_train, train_meta_pred)*100))
print("   Classification Report:")
print(classification_report(y_meta_train, train_meta_pred))
print()

X_meta_test = final_test[meta_features]
y_meta_test = final_test["progress_status"]

meta_test_pred = meta_model.predict(X_meta_test)
print("Final Meta-Model (on Testing Data):")
print("   Accuracy: {:.2f}%".format(accuracy_score(y_meta_test, meta_test_pred)*100))
print("   Classification Report:")
print(classification_report(y_meta_test, meta_test_pred))
print()

cm = confusion_matrix(y_meta_test, meta_test_pred, labels=[1, 2])
if cm.shape == (2,2):
    error_rate_2_when_should_1 = (cm[0, 1] / cm[0, :].sum()) * 100 if cm[0, :].sum() > 0 else np.nan
    error_rate_1_when_should_2 = (cm[1, 0] / cm[1, :].sum()) * 100 if cm[1, :].sum() > 0 else np.nan

    print("Misclassification Rates on Test Data:")
    print("   When true class is 1, predicted as 2: {:.2f}%".format(error_rate_2_when_should_1))
    print("   When true class is 2, predicted as 1: {:.2f}%".format(error_rate_1_when_should_2))
else:
    print("Confusion matrix did not have expected shape for classes [1, 2].")

final_test["Final_Model_Prediction"] = meta_test_pred
final_test.to_excel("final_model_predictions.xlsx", index=False)

print("Pipeline complete. Final meta-model predictions saved to 'final_model_predictions.xlsx'.")
