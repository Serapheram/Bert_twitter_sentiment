import pandas as pd
import random

positive = [
    "I love the new update, it works really well.",
    "Amazing support team, very helpful!",
    "Best purchase I made this year, highly recommend.",
    "This app makes my life so much easier.",
    "Great experience, will buy again.",
    "So happy with this product, exceeded expectations.",
    "The delivery was fast and packaging was perfect.",
    "Customer service resolved my issue instantly, love it.",
    "This feature is a game changer, awesome work.",
    "Really impressed with the quality, five stars."
]

negative = [
    "This app is slow and keeps crashing.",
    "Worst experience with delivery today.",
    "Terrible customer support, no one replied.",
    "The product broke after one day, waste of money.",
    "Very disappointed with this update.",
    "Awful experience, will not buy again.",
    "This is the worst app I have ever used.",
    "Battery drains so fast, very annoying.",
    "Completely useless, do not recommend.",
    "Late delivery and damaged item, so frustrating."
]

neutral = [
    "The service was okay, nothing special.",
    "It works as expected, nothing more.",
    "The product arrived on time.",
    "I have mixed feelings about this update.",
    "It is an average app, does the job.",
    "Not bad, not great, just okay.",
    "The price is reasonable for what you get.",
    "Delivery took the usual time.",
    "The app has some features I use occasionally.",
    "It is fine, no strong opinion either way."
]

rows = []
for _ in range(150):
    rows.append((random.choice(positive), "positive"))
    rows.append((random.choice(negative), "negative"))
    rows.append((random.choice(neutral), "neutral"))

random.shuffle(rows)
df = pd.DataFrame(rows, columns=["text", "sentiment"])
df.to_csv("dataset/twitter_sentiment.csv", index=False)
print("Saved", len(df), "rows")