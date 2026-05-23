from sklearn.feature_extraction.text import TfidfVectorizer
from .cleanup import clean_value

def extract_user_keywords(interactions, top_n=30):
    texts = []

    for _, row in interactions.iterrows():
        comment = clean_value(row["comment"])
        title = clean_value(row["title"])
        subreddit = clean_value(row["subreddit"])
        text = f"{comment} {title} {subreddit}".strip()
        if text:
            texts.append(text)

    if not texts:
        return []
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=800,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.70,
        token_pattern=r"(?u)\\b[a-zA-Z][a-zA-Z0-9_]{2,}\\b"
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1

    keyword_scores = list(zip(feature_names, scores))
    keyword_scores = sorted(keyword_scores, key=lambda item: item[1], reverse=True)
    
    keywords = []

    for keyword, score in keyword_scores:
        keyword = keyword.strip().lower()

        if len(keyword) < 3:
            continue

        if keyword.isdigit():
            continue

        keywords.append(keyword)

        if len(keywords) >= top_n:
            break
        
    return keywords

def get_keyword_bonus(post_title, post_body, user_keywords):
    import re
    post_text = f"{post_title} {post_body}".lower()

    if not user_keywords:
        return 0.0
    
    matches = 0

    for keyword in user_keywords:
        keyword = keyword.lower().strip()

        if not keyword:
            continue

        pattern = r"\\b" + re.escape(keyword) + r"\\b"
        
        if re.search(pattern, post_text):
            matches += 1

    bonus = matches * 0.005
    return min(bonus, 0.04)
