import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
import numpy as np

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    data = []
    for col in df.columns:
        if col.endswith('_label'):
            value_col = col[:-6].strip()
            for value, label in zip(df[value_col], df[col]):
                data.append({
                    "value": str(value),
                    "label": 1 if label.lower() == "true" else 0
                })
    return pd.DataFrame(data)

class PIIDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_deep_pii_model(csv_path, model_path="pii_bert"):
    df = load_and_prepare_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(df["value"], df["label"], test_size=0.2, random_state=42)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = PIIDataset(list(X_train), list(y_train), tokenizer)
    test_dataset = PIIDataset(list(X_test), list(y_test), tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model and tokenizer saved to {model_path}")

def predict_pii_bert(input_array, model_path="pii_bert"):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    inputs = tokenizer(input_array, truncation=True, padding=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).numpy()
    return preds


def evaluate_model_on_csv(csv_path, model_path="pii_bert"):
    df = load_and_prepare_data(csv_path)
    y_true = np.array(df["label"])
    y_pred = predict_pii_bert(list(df["value"]), model_path=model_path)

    # Calculate confusion matrix components
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    total = TP + TN + FP + FN

    print(f"True Positive (TP): {TP} ({TP/total:.2%})")
    print(f"True Negative (TN): {TN} ({TN/total:.2%})")
    print(f"False Positive (FP): {FP} ({FP/total:.2%})")
    print(f"False Negative (FN): {FN} ({FN/total:.2%})")

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "TP%": TP/total, "TN%": TN/total, "FP%": FP/total, "FN%": FN/total
    }

if __name__ == "__main__":

    model_path = "pii_bert"
    if not os.path.exists(model_path):
        train_deep_pii_model("../data/pii_synthetic_dataset_gpt2_labeled_modified.csv", model_path=model_path)


    test_data = [
        "John Doe",
        "123-45-6789",
        "info@example.com",
        "123 Main St, London",
        "Engineer",
        "null",
        "nil",
        "none",
        "aman",
        "0",
        "device-001",
        "6226-4124-6552-9576",
        "6226412465529576",
        "6226 4124 6552 9576",
        "+91 7091071590"
    ]
    results = predict_pii_bert(test_data)
    result_map = {k: int(v) for k, v in zip(test_data, results)}
    print(result_map)
    for val, is_pii in zip(test_data, results):
        print(f"{val!r} => {'PII' if is_pii else 'Not PII'}")

    # print("\nEvaluating on full labeled dataset:")
    # evaluate_model_on_csv("../data/fusion_matrix-dataset.csv")





















# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.base import BaseEstimator, TransformerMixin
# import joblib
# import re

# def normalize_card_number(val):
#     if isinstance(val, str):
#         digits = re.sub(r"[\s\-]", "", val)
#         if re.fullmatch(r"\d{16}", digits):
#             return digits
#     return val

# class CardNumberFlagger(BaseEstimator, TransformerMixin):
#     """Custom transformer to flag card number patterns."""
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         # Returns a column vector: 1 if value is a card number, else 0
#         return [[1 if re.fullmatch(r"\d{16}", re.sub(r"[\s\-]", "", str(x))) else 0] for x in X]

# def load_and_prepare_data(csv_path):
#     df = pd.read_csv(csv_path, dtype=str).fillna("")
#     data = []
#     for col in df.columns:
#         if col.endswith('_label'):
#             value_col = col[:-6].strip()
#             for value, label in zip(df[value_col], df[col]):
#                 # Normalize card numbers for relevant columns
#                 if value_col in ["credit card number", "debit card number"]:
#                     value = normalize_card_number(value)
#                 data.append({
#                     "value": str(value),
#                     "column": value_col,
#                     "label": label.lower() == "true"
#                 })
#     return pd.DataFrame(data)

# def train_pii_model(csv_path, model_path="pii_model.joblib"):
#     data = load_and_prepare_data(csv_path)
#     X = data["value"]
#     y = data["label"]

#     # Combine TF-IDF and card number flag
#     features = FeatureUnion([
#         ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=1000)),
#         ("card_flag", CardNumberFlagger())
#     ])

#     pipeline = Pipeline([
#         ("features", features),
#         ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
#     ])

#     pipeline.fit(X, y)
#     joblib.dump(pipeline, model_path)
#     print(f"Model trained and saved to {model_path}")

# def predict_pii(input_array, model_path="pii_model.joblib"):
#     pipeline = joblib.load(model_path)
#     normalized = [normalize_card_number(x) for x in input_array]
#     preds = pipeline.predict(normalized)
#     return preds

# if __name__ == "__main__":
#     train_pii_model("../../pii_synthetic_dataset_gpt2_labeled_modified.csv")

#     test_data = [
#         "John Doe",
#         "123-45-6789",
#         "info@example.com",
#         "123 Main St",
#         "Engineer",
#         "null",
#         "nil",
#         "none",
#         "aman",
#         "0",
#         "device-001",
#         "6226-4124-6552-9576",
#         "6226412465529576",
#         "6226 4124 6552 9576",
#         "+91 7091071590"
#     ]
#     results = predict_pii(test_data)
#     for val, is_pii in zip(test_data, results):
#         print(f"{val!r} => {'PII' if is_pii else 'Not PII'}")