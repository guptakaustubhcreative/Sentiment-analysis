import re
positive_words = {
"good", "great", "excellent", "amazing", "love", "nice", "happy", "awesome"
}
negative_words = {
    "bad", "worst", "terrible", "hate", "poor", "sad", "awful"
}
negation_words = {"not", "no", "never", "n't"}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()
import re
def analyze_sentiment(text):
    words = preprocess(text)
    sentiment_score = 0
    negation = False

    for word in words:
        if word in negation_words:
            negation = True
        elif word in positive_words:
            sentiment_score += -1 if negation else 1
            negation = False
        elif word in negative_words:
            sentiment_score += 1 if negation else -1
            negation = False

    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"