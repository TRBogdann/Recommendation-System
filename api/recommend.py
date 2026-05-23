
import numpy as np

from api.cleanup import clean_value
from api.collaborative import get_collaborative_score, get_similarity_df, is_collaborative_enabled
from api.embeddings import get_embedding, get_faiss_index, get_post_embeddings, get_posts_df
from api.keywords import extract_user_keywords, get_keyword_bonus
from api.profiling import build_user_profile_text, load_user_interactions
from api.sentiment import analyze_text_sentiment, get_sentiment_bonus, get_sentiment_pipeline, infer_user_dominant_sentiment
from api.subreddit import extract_user_subreddits, get_same_subreddit_candidate_indices, get_subreddit_bonus
from api.llm import generate_recommendation_explanation

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
