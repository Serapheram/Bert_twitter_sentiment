import sys
import pandas as pd
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QLabel, QLineEdit, QFileDialog, QMessageBox
)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

class SentimentApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.df = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Twitter Sentiment Analysis - BERT PyQt GUI")
        self.resize(800, 500)

        layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.load_data_btn = QPushButton("Load Dataset CSV")
        self.load_data_btn.clicked.connect(self.load_dataset)
        self.load_model_btn = QPushButton("Load BERT Model")
        self.load_model_btn.clicked.connect(self.load_model)
        btn_layout.addWidget(self.load_data_btn)
        btn_layout.addWidget(self.load_model_btn)
        layout.addLayout(btn_layout)

        self.status_label = QLabel("Status: nothing loaded yet")
        layout.addWidget(self.status_label)

        self.table = QTableWidget()
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["Tweet Text"])
        self.table.cellClicked.connect(self.predict_from_table)
        layout.addWidget(self.table)

        manual_layout = QHBoxLayout()
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type a sentence to predict...")
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.predict_manual)
        manual_layout.addWidget(self.text_input)
        manual_layout.addWidget(self.predict_btn)
        layout.addLayout(manual_layout)

        self.result_label = QLabel("Predicted Sentiment: -")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load CSV", "", "CSV Files (*.csv)")
        if path:
            self.df = pd.read_csv(path)
            text_col = "text" if "text" in self.df.columns else self.df.columns[0]
            self.table.setRowCount(len(self.df))
            for i, row in enumerate(self.df[text_col]):
                self.table.setItem(i, 0, QTableWidgetItem(str(row)))
            self.status_label.setText(f"Dataset loaded: {len(self.df)} rows")

    def load_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select saved_bert_model folder")
        if path:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(path)
            self.model = DistilBertForSequenceClassification.from_pretrained(path)
            self.model.eval()
            self.status_label.setText(f"Model loaded from {path}")

    def predict(self, text):
        if not self.model or not self.tokenizer:
            QMessageBox.warning(self, "No model", "Load the BERT model first.")
            return
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            conf = torch.softmax(outputs.logits, dim=1).max().item()
        self.result_label.setText(f"Predicted Sentiment: {label_map[pred]}  ({conf*100:.1f}% confidence)")

    def predict_from_table(self, row, col):
        text = self.table.item(row, col).text()
        self.text_input.setText(text)
        self.predict(text)

    def predict_manual(self):
        text = self.text_input.text().strip()
        if text:
            self.predict(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SentimentApp()
    window.show()
    sys.exit(app.exec_())