import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv("dataset/twitter_sentiment.csv")
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].map(label_map)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_texts, train_labels)
val_dataset = TweetDataset(val_texts, val_labels)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

args = TrainingArguments(
    output_dir="results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

train_result = trainer.train()
eval_result = trainer.evaluate()
print(eval_result)

# save model
model.save_pretrained("saved_bert_model")
tokenizer.save_pretrained("saved_bert_model")

# graphs
log_history = trainer.state.log_history
train_loss = [x["loss"] for x in log_history if "loss" in x]
eval_loss = [x["eval_loss"] for x in log_history if "eval_loss" in x]
eval_acc = [x["eval_accuracy"] for x in log_history if "eval_accuracy" in x]

plt.figure()
plt.plot(train_loss, label="train loss")
plt.plot(eval_loss, label="val loss")
plt.legend(); plt.title("Loss"); plt.savefig("loss_graph.png")

plt.figure()
plt.plot(eval_acc, label="val accuracy")
plt.legend(); plt.title("Accuracy"); plt.savefig("accuracy_graph.png")

preds = trainer.predict(val_dataset)
y_pred = preds.predictions.argmax(-1)
cm = confusion_matrix(val_labels, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title("Confusion Matrix"); plt.savefig("confusion_matrix.png")

plt.figure()
df["sentiment"].value_counts().plot(kind="bar")
plt.title("Class Distribution"); plt.savefig("class_distribution.png")

print("Done. Model saved to saved_bert_model/")