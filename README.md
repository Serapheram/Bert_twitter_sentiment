# Twitter Sentiment Analysis using BERT and PyQt GUI

## Model
DistilBERT (distilbert-base-uncased) fine-tuned for 3-class sentiment classification (positive, negative, neutral).
Train/test split: 80/20. Epochs: 3. Batch size: 16.

## Results
Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1-score: 1.00

## Graphs
<img width="640" height="480" alt="loss_graph" src="https://github.com/user-attachments/assets/f6f75414-7624-4702-b655-15a34e6f37d3" />
<img width="640" height="480" alt="accuracy_graph" src="https://github.com/user-attachments/assets/ea8131dd-8de6-4793-8b5b-b8fd5eec11d5" />
<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/b0764fe2-c510-42c3-9b14-8736a95f34f0" />
<img width="640" height="480" alt="class_distribution" src="https://github.com/user-attachments/assets/6b8fa753-651f-43c2-b318-4b094fc81964" />


## Paul's Critical Thinking Standards
- Clarity: Dataset has two columns, text (tweet) and sentiment (positive/negative/neutral). GUI lets user load dataset, load model, click a tweet, and see prediction.
- Accuracy: Model reached 100% accuracy on the validation set.
- Precision: Model used is distilbert-base-uncased, trained for 3 epochs, batch size 16, 80/20 train/test split.
- Relevance: Confusion matrix and accuracy/loss graphs directly reflect model performance on this dataset.
- Depth: 100% accuracy is because the dataset is small and synthetically generated with clearly separated vocabulary per class; a larger, real-world Twitter dataset would show lower and more realistic accuracy.
- Logic: Training, saving, loading, and GUI prediction were tested end to end and work correctly.
- Fairness: Dataset is synthetic and may not reflect real slang, sarcasm, or class imbalance found in real tweets. Real-world performance would likely be lower.

## Dataset Source
Synthetically generated for this assignment (see make_dataset.py), inspired by common Twitter sentiment phrasing patterns.

## GitHub Repository
https://github.com/Serapheram/Bert_twitter_sentiment
