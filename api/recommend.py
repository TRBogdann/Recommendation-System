
import numpy as np

from api.cleanup import clean_value
from api.collaborative import get_collaborative_score, get_similarity_df, is_collaborative_enabled
from api.embeddings import get_embedding, get_faiss_index, get_post_embeddings, get_posts_df
from api.keywords import extract_user_keywords, get_keyword_bonus
from api.profiling import build_user_profile_text, load_user_interactions
from api.sentiment import analyze_text_sentiment, get_sentiment_bonus, get_sentiment_pipeline, infer_user_dominant_sentiment
from api.subreddit import extract_user_subreddits, get_same_subreddit_candidate_indices, get_subreddit_bonus
from api.llm import generate_recommendation_explanation



# def get_collaborative_score(
#     candidate_post_id,
#     interacted_post_ids,
#     similarity_df=None
# ):
#     if not USE_COLLABORATIVE: 
#         return 0.0 
    
#     if similarity_df is None: 
#         return 0.0
    
#     if candidate_post_id not in similarity_df.index:
#         return 0.0

#     similarities = []

#     for interacted_post_id in interacted_post_ids:
#         if interacted_post_id not in similarity_df.index:
#             continue

#         similarity = similarity_df.loc[
#             candidate_post_id,
#             interacted_post_id
#         ]

#         similarities.append(similarity)

#     if not similarities:
#         return 0.0

#     return float(np.mean(similarities))


# def clean_value(value):
#     if pd.isna(value):
#         return ""

#     value = str(value).strip()

#     if value.lower() in ["nan", "none", "null", "[deleted]", "[removed]"]:
#         return ""

#     return value





# def build_user_profile_text(interactions, user_keywords=None, user_subreddits=None):
#     if interactions.empty:
#         return None

#     username = interactions["username"].iloc[0]

#     comments = " ".join(
#         clean_value(comment)
#         for comment in interactions["comment"].tolist()
#     )

#     titles = " ".join(
#         clean_value(title)
#         for title in interactions["title"].dropna().unique().tolist()
#     )

#     subreddits_text = ", ".join(sorted(user_subreddits)) if user_subreddits else ""

#     keywords_text = ", ".join(user_keywords) if user_keywords else ""

#     profile_text = f"""
# User profile generated from Reddit interactions.
# Username: {username}

# The user frequently interacts with these subreddits:
# {subreddits_text}

# Important terms automatically extracted from the user's activity:
# {keywords_text}

# The user's comments:
# {comments}

# Titles of posts the user interacted with:
# {titles}

# Recommend Reddit posts that match the user's interests, communities, terminology, topics, and previous activity.
# """.strip()

#     return profile_text


# def extract_user_subreddits(interactions):
#     return set(
#         clean_value(subreddit).lower()
#         for subreddit in interactions["subreddit"].tolist()
#         if clean_value(subreddit)
#     )


# def extract_user_keywords(interactions, top_n=30):
#     texts = []

#     for _, row in interactions.iterrows():
#         comment = clean_value(row["comment"])
#         title = clean_value(row["title"])
#         subreddit = clean_value(row["subreddit"])

#         text = f"{comment} {title} {subreddit}".strip()

#         if text:
#             texts.append(text)

#     if not texts:
#         return []

#     vectorizer = TfidfVectorizer(
#         stop_words="english",
#         max_features=800,
#         ngram_range=(1, 2),
#         min_df=2,
#         max_df=0.70,
#         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{2,}\b"
#     )

#     try:
#         tfidf_matrix = vectorizer.fit_transform(texts)
#     except ValueError:
#         return []

#     feature_names = vectorizer.get_feature_names_out()
#     scores = tfidf_matrix.sum(axis=0).A1

#     keyword_scores = list(zip(feature_names, scores))
#     keyword_scores = sorted(keyword_scores, key=lambda item: item[1], reverse=True)

#     keywords = []

#     for keyword, score in keyword_scores:
#         keyword = keyword.strip().lower()

#         if len(keyword) < 3:
#             continue

#         if keyword.isdigit():
#             continue

#         keywords.append(keyword)

#         if len(keywords) >= top_n:
#             break

#     return keywords


# def map_sentiment_label(label):
#     label = str(label).lower()

#     if "positive" in label or label == "label_2":
#         return "positive"

#     if "negative" in label or label == "label_0":
#         return "negative"

#     return "neutral"


# def analyze_text_sentiment(text, sentiment_pipeline):
#     if sentiment_pipeline is None:
#         return "neutral", 0.0
    
#     text = clean_value(text)

#     if text == "":
#         return "neutral", 0.0

#     result = sentiment_pipeline(text[:512])[0]

#     sentiment = map_sentiment_label(result["label"])
#     confidence = float(result["score"])

#     return sentiment, confidence


# def infer_user_dominant_sentiment(interactions, sentiment_pipeline):
#     sentiments = []

#     for _, row in interactions.iterrows():
#         comment = clean_value(row["comment"])
#         title = clean_value(row["title"])

#         text = f"{comment}. {title}"

#         if text.strip() == ".":
#             continue

#         sentiment, confidence = analyze_text_sentiment(text, sentiment_pipeline)
#         sentiments.append(sentiment)

#     if not sentiments:
#         return "neutral", Counter()

#     counts = Counter(sentiments)
#     dominant_sentiment = counts.most_common(1)[0][0]

#     return dominant_sentiment, counts


# def get_sentiment_bonus(post_sentiment, user_dominant_sentiment):
#     if user_dominant_sentiment == "neutral":
#         if post_sentiment == "neutral":
#             return 0.01
#         return 0.0

#     if post_sentiment == user_dominant_sentiment:
#         return 0.03

#     if post_sentiment == "neutral":
#         return 0.01

#     return -0.02


# def get_subreddit_bonus(post_subreddit, user_subreddits):
#     post_subreddit = clean_value(post_subreddit).lower()

#     if post_subreddit in user_subreddits:
#         return 0.15

#     return 0.0


# def get_keyword_bonus(post_title, post_body, user_keywords):
#     post_text = f"{post_title} {post_body}".lower()

#     if not user_keywords:
#         return 0.0

#     matches = 0

#     for keyword in user_keywords:
#         keyword = keyword.lower().strip()

#         if not keyword:
#             continue

#         pattern = r"\b" + re.escape(keyword) + r"\b"

#         if re.search(pattern, post_text):
#             matches += 1

#     bonus = matches * 0.005

#     return min(bonus, 0.04)


# def get_same_subreddit_candidate_indices(posts_df, user_subreddits, max_per_subreddit=100):
#     candidate_indices = []

#     for subreddit in user_subreddits:
#         matching_rows = posts_df[
#             posts_df["subreddit"].fillna("").str.lower() == subreddit
#         ]

#         if matching_rows.empty:
#             continue

#         candidate_indices.extend(matching_rows.index.tolist()[:max_per_subreddit])

#     return candidate_indices


def recommend_posts_for_user(user_id, top_x=10):
    interactions = load_user_interactions(user_id)

    if interactions.empty:
        return None, []

    username = clean_value(interactions["username"].iloc[0])

    user_subreddits = extract_user_subreddits(interactions)
    user_keywords = extract_user_keywords(interactions, top_n=30)

    user_profile_text = build_user_profile_text(
        interactions,
        user_keywords=user_keywords,
        user_subreddits=user_subreddits
    )

    sentiment_pipeline = get_sentiment_pipeline()

    user_dominant_sentiment, sentiment_counts = infer_user_dominant_sentiment(
        interactions,
        sentiment_pipeline
    )

    posts_df = get_posts_df()
    similarity_df = get_similarity_df()
    index = get_faiss_index()

    user_embedding = get_embedding(user_profile_text)

    candidate_count = max(top_x * 20, 200)

    semantic_scores, candidate_indices = index.search(
        user_embedding,
        candidate_count
    )

    global_candidate_indices = candidate_indices[0].tolist()

    same_subreddit_indices = get_same_subreddit_candidate_indices(
        posts_df,
        user_subreddits,
        max_per_subreddit=150
    )

    all_candidate_indices = list(
        dict.fromkeys(global_candidate_indices + same_subreddit_indices)
    )

    interacted_post_ids = set(
        interactions["post_id"].dropna().tolist()
    )

    results = []

    post_embeddings = get_post_embeddings()

    for rank, idx in enumerate(all_candidate_indices, start=1):
        post = posts_df.iloc[idx]
        post_id = post["id"]

        collaborative_score = get_collaborative_score(
            post_id,
            interacted_post_ids,
            similarity_df
        )

        title = clean_value(post["title"])
        subreddit = clean_value(post["subreddit"])
        body = clean_value(post["body"])

        post_text_for_sentiment = f"{title}. {body}"

        post_sentiment, post_sentiment_confidence = analyze_text_sentiment(
            post_text_for_sentiment,
            sentiment_pipeline
        )

        post_embedding = post_embeddings[idx]
        semantic_score = float(np.dot(user_embedding[0], post_embedding))

        subreddit_bonus = get_subreddit_bonus(
            subreddit,
            user_subreddits
        )

        keyword_bonus = get_keyword_bonus(
            title,
            body,
            user_keywords
        )

        sentiment_bonus = get_sentiment_bonus(
            post_sentiment,
            user_dominant_sentiment
        )

        if is_collaborative_enabled(): 
            final_score = ( 
                0.65 * semantic_score + 
                0.2 * collaborative_score + 
                subreddit_bonus + 
                keyword_bonus + 
                sentiment_bonus ) 
        else: 
            final_score = (
                0.85 * semantic_score + 
                subreddit_bonus + 
                keyword_bonus + 
                sentiment_bonus )
            
        context = {
            "user_keywords": user_keywords,
            "user_subreddits": user_subreddits,
            "recommended_title": title,
            "recommended_subreddit": subreddit,
            "semantic_score": semantic_score,
            "collaborative_score": collaborative_score
        }

        results.append({
            "rank": 0,
            "id": post_id,
            "title": title,
            "subreddit": subreddit,
            "body": body[:300],
            "semantic_score": semantic_score,
            "collaborative_score": collaborative_score,
            "subreddit_bonus": subreddit_bonus,
            "keyword_bonus": keyword_bonus,
            "post_sentiment": post_sentiment,
            "post_sentiment_confidence": post_sentiment_confidence,
            "user_dominant_sentiment": user_dominant_sentiment,
            "sentiment_bonus": sentiment_bonus,
            "final_score": final_score,
            "llm_context": context
        })

    results = sorted(
        results,
        key=lambda item: item["final_score"],
        reverse=True
    )

    results = results[:top_x]

    for i, result in enumerate(results, start=1):
        result["rank"] = i
        result["id"] = str(result["id"])

        try:
            explanation = generate_recommendation_explanation( 
                user_keywords=result["llm_context"]["user_keywords"], 
                user_subreddits=result["llm_context"]["user_subreddits"], 
                recommended_title=result["llm_context"]["recommended_title"], 
                recommended_subreddit=result["llm_context"]["recommended_subreddit"], 
                semantic_score=result["llm_context"]["semantic_score"], 
                collaborative_score=result["llm_context"]["collaborative_score"]
            )
            result["explanation"] = explanation
        except Exception as e:
            print(f"Error generating explanation for post_id={result['id']}: {e}")
            result["explanation"] = "No explanation available."
        finally:
            result.pop("llm_context", None)

    return username, results
