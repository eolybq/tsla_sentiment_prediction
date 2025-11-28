import pandas as pd
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

df = pd.read_csv("data/rawdata/tweets_tsla_daily.csv")
df_not_agr = pd.read_csv("/data/rawdata/tweets_tsla_not_agregated.csv")

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
pipe = pipeline("text-classification", model="ProsusAI/finbert", tokenizer=tokenizer, top_k = None)


def get_sentiment_probs(text):
    if pd.isna(text) or not str(text).strip():
        return {"positive": None, "neutral": None, "negative": None, "label": "NEUTRAL"}
    tokens = tokenizer.encode(str(text), max_length=512, truncation=True, return_tensors="pt")
    truncated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    result = pipe(truncated_text)[0]
    # Výsledek je seznam slovníků, např. [{'label': 'positive', 'score': ...}, ...]
    scores = {item["label"]: item["score"] for item in result}
    # Najdi label s nejvyšším skóre
    max_label = max(scores, key=scores.get)
    scores["label"] = max_label
    return scores


tqdm.pandas()

sentiments = df["cleanText"].progress_apply(get_sentiment_probs)
df["sentiment"] = sentiments.apply(lambda x: x["label"])
df["sentiment_positive"] = sentiments.apply(lambda x: x["positive"])
df["sentiment_neutral"] = sentiments.apply(lambda x: x["neutral"])
df["sentiment_negative"] = sentiments.apply(lambda x: x["negative"])

sentiments_not_agr = df_not_agr["cleanText"].progress_apply(get_sentiment_probs)
df_not_agr["sentiment"] = sentiments_not_agr.apply(lambda x: x["label"])
df_not_agr["sentiment_positive"] = sentiments_not_agr.apply(lambda x: x["positive"])
df_not_agr["sentiment_neutral"] = sentiments_not_agr.apply(lambda x: x["neutral"])
df_not_agr["sentiment_negative"] = sentiments_not_agr.apply(lambda x: x["negative"])


df.to_csv("data/rawdata/tweets_tsla_daily_sentiment.csv", index=False)
df_not_agr.to_csv("data/rawdata/tweets_tsla_not_agregated_sentiment.csv", index=False)
