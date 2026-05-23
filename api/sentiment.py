from collections import Counter
import os

from dotenv import load_dotenv
from .cleanup import clean_value

load_dotenv()
_SENTIMENT_PIPELINE = None

def get_sentiment_pipeline():
    global _SENTIMENT_PIPELINE

    USE_SENTIMENT = os.getenv("USE_SENTIMENT", "true").lower() == "true"
    if not USE_SENTIMENT:
        print("Sentiment analysis is disabled via environment variable.")
        return None

    if _SENTIMENT_PIPELINE is None:
        from transformers import pipeline
        _SENTIMENT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

    return _SENTIMENT_PIPELINE

def map_sentiment_label(label):
    label = str(label).lower()
    if "positive" in label or label == "label_2":
        return "positive"
    if "negative" in label or label == "label_0":
        return "negative"
    return "neutral"

def analyze_text_sentiment(text, sentiment_pipeline):
    if sentiment_pipeline is None:
        return "neutral", 0.0
    
    text = clean_value(text)

    if text == "":
        return "neutral", 0.0
    
    result = sentiment_pipeline(text[:512])[0]

    sentiment = map_sentiment_label(result["label"])
    confidence = float(result["score"])

    return sentiment, confidence

def infer_user_dominant_sentiment(interactions, sentiment_pipeline):
    sentiments = []

    for _, row in interactions.iterrows():
        comment = clean_value(row["comment"])
        title = clean_value(row["title"])

        text = f"{comment}. {title}"

        if text.strip() == ".":
            continue

        sentiment, confidence = analyze_text_sentiment(text, sentiment_pipeline)
        sentiments.append(sentiment)

    if not sentiments:
        return "neutral", Counter()
    
    counts = Counter(sentiments)
    dominant_sentiment = counts.most_common(1)[0][0]
    return dominant_sentiment, counts

def get_sentiment_bonus(post_sentiment, user_dominant_sentiment):
    if user_dominant_sentiment == "neutral":
        if post_sentiment == "neutral":
            return 0.01
        return 0.0
    if post_sentiment == user_dominant_sentiment:
        return 0.03
    if post_sentiment == "neutral":
        return 0.01
    return -0.02
